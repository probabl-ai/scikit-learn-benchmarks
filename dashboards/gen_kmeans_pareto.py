"""`configs/kmeans_pareto.py`: fit time against inertia of k-means implementations, one
Pareto front per implementation and machine, one plot per (dataset, n_clusters).

Every trace carries its implementation and machine in `meta`, and the checkboxes above
the plots hide the traces of the unchecked ones in the browser. The URL hash records
the checked boxes of each group that differs from the defaults, so filtered views can
be shared.
"""

from collections import defaultdict
from functools import cache
from html import escape
import math
from pathlib import Path
import re
import statistics

from plotly import graph_objects as go

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import read_env, software_build_name
from sklbench.reporting.html import BASE_TEMPLATE, DATE_RANGE_TEMPLATE, PLOT_TEMPLATE
from sklbench.reporting.matching import (
    BenchmarkRecord,
    date_range,
    matches_source_configs,
    read_benchmark_records,
)
from sklbench.reporting.utils import stable_json, without_keys


TITLE = "k-means: fit time against inertia"
SOURCE_CONFIGS = [
    "configs/kmeans_pareto.py",
    "configs/kmeans_pareto_reference.py",
    # Temporary, until the full sweep runs.
    "configs/kmeans_pareto_fast.py",
]
REFERENCE_CONFIG = "configs/kmeans_pareto_reference.py"
SOURCE_ENVS = [
    "sklearn-dev",
    "faiss-cpu",
    "faiss-gpu",
    "cuml",
    "skl-nvidia",
    "sklearn-contrib-emeans",
    "sklearn-contrib-emeans-mkl",
    "sklearn-contrib-emeans-xpu",
    "sklearn-contrib-emeans-mps",
    "emeans-rs",
]
OUTPUT = "kmeans_pareto.html"

DATASET_NAMES = {
    "glove_200": "GloVe-200",
    "sift_128": "SIFT1M",
    "clip_512": "CLIP ViT-B/32",
    "susy": "SUSY",
    "year_prediction_msd_standardized": "YearPredictionMSD (standardized)",
    "road_network_km_x10": "3D Road Network (10x oversampled)",
    "20newsgroups": "20 Newsgroups (sparse)",
    "rcv1": "RCV1 (sparse)",
}

# One color per implementation, the same on every plot: scikit-learn's cyan and orange
# for scikit-learn Lloyd and sklearn-contrib-emeans, Elkan in a darker blue, the others
# picked to stay apart for color-blind readers too.
IMPLEMENTATIONS = {
    # label: (color, device)
    "scikit-learn Lloyd": ("#2294c4", "CPU"),
    "scikit-learn Elkan": ("#15688c", "CPU"),
    "sklearn-contrib-emeans": ("#f7931e", "CPU"),
    "FAISS e-means (NumPy)": ("#089868", "CPU"),
    "emeans (Rust)": ("#805bd5", "CPU"),
    "sklearn-contrib-emeans (PyTorch)": ("#b35806", "GPU"),
    "cuML KMeans": ("#9c0451", "GPU"),
    "FAISS Lloyd (GPU)": ("#ea5974", "GPU"),
    "FAISS e-means (Triton, FP32)": ("#4041b6", "GPU"),
    "FAISS e-means (BF16)": ("#c075eb", "GPU"),
}
# Still benchmarked, and their runs count for the best known inertia, but not plotted.
HIDDEN_IMPLEMENTATIONS = {
    "scikit-learn EMeans",
    "FAISS Lloyd (CPU)",
    "FAISS e-means (Cython kernel)",
    "CuPy k-means",
    "CuPy k-means (fused kernels)",
    "emeans (Rust)",
}
# The boxes checked when the URL doesn't say otherwise.
DEFAULT_IMPLEMENTATIONS = {
    "scikit-learn Lloyd",
    "sklearn-contrib-emeans",
    "FAISS e-means (NumPy)",
    "sklearn-contrib-emeans (PyTorch)",
    "cuML KMeans",
    "FAISS e-means (Triton, FP32)",
}
DEFAULT_MACHINES = {"332097", "d189c4"}  # Intel Core Ultra desktop, NVIDIA L4 VM
MACHINE_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "star"]
# Relative inertias below this tie with the best in the hover: runs that reach the
# same centers on two machines differ by ~1e-7 from floating-point rounding.
TIED_WITH_BEST = 1e-6
# Relative inertias below this are drawn on a "best known inertia" line at this
# value, the same on every plot.
BEST_LINE = 1e-5
PLOT_FOOTNOTE = (
    "median over seeds; whiskers: min to max over seeds; faded: off the front; "
    "hover a point for its settings"
)
# Parameters that pick the implementation rather than a setting.
IMPLEMENTATION_PARAMS = {"algorithm", "fused", "gpu", "n_clusters"}

ABOUT_HTML = """<section class="panel">
  <p>For a given time budget, which k-means implementation and settings give the lowest
  <dfn>inertia</dfn>? Each plot shows the <dfn term="fit">fit</dfn> time against the
  inertia of the returned centers, for one dataset and one number of clusters, with
  one <dfn>Pareto front</dfn> per implementation and machine, as
  <a href="https://ann-benchmarks.com">ann-benchmarks</a> does for nearest neighbor
  search. Lower left is better.</p>
  <p>Comparing the time of a Lloyd iteration is not enough: inits differ in cost and
  quality (random, <dfn>k-means++</dfn>, <dfn>k-means||</dfn>), and
  <dfn>e-means</dfn> trades iterations for quality differently. The settings of each
  implementation (init, number of inits, iterations or epochs, tolerance) sweep the
  time budget.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>A point is one setting of one implementation: the median over 3 seeds of the
    fit time and of the <dfn>relative inertia</dfn>, with whiskers from the best to the
    worst seed. The line is the Pareto front of the implementation on that machine:
    with a time budget t, the best it gets is the last step left of t. Faded points are
    off the front. Hover a point for its settings, iterations and empty clusters.</p>
    <p>The fit time includes the init, with the data already in host memory for CPU
    implementations and in device memory for GPU ones (except FAISS on the GPU, whose
    training loop runs on the host). The inertia is the mean squared distance of every
    sample to its nearest center, computed in float64 from the returned centers by the
    same code for every implementation. Everything runs in float32 with TF32 off, and
    CPU implementations get the same number of threads: one per physical core, at
    most 64. Machines with more cores run on their first 64 physical cores.</p>
    <p>The best known inertia of a dataset and number of clusters is the lowest of any
    run on any machine, including long reference runs, implementations and builds that are not
    plotted. Points and whiskers on the dotted line at 1e-5 are within 1e-5 of it, and
    the hover gives their exact relative inertia, or "best" below 1e-6 (rounding
    differences between machines are ~1e-7).</p>
  </details>
  <details class="about-section">
    <summary>Implementations</summary>
    <ul>
      <li>scikit-learn Lloyd: <code>KMeans(algorithm="lloyd")</code> of scikit-learn
      main, which includes the faster k-means++ of
      <a href="https://github.com/scikit-learn/scikit-learn/pull/35140">#35140</a>.</li>
      <li>scikit-learn Elkan: <code>KMeans(algorithm="elkan")</code>, only on the 3D
      Road Network, since it only pays off in low dimension.</li>
      <li>sklearn-contrib-emeans: <code>EMeans</code> of
      <a href="https://pypi.org/project/sklearn-contrib-emeans/">sklearn-contrib-emeans</a>
      0.2.1, e-means in Cython, with a random or k-means|| init. Its GEMMs go
      through SciPy's BLAS: MKL (conda-forge NumPy and SciPy) on the Intel CPUs, and
      OpenBLAS (PyPI wheels) on the others.</li>
      <li>sklearn-contrib-emeans (PyTorch): the same <code>EMeans</code> (0.3.0) on
      PyTorch tensors on the GPU, through the array API
      (<code>array_api_dispatch=True</code>): XPU on the Intel GPUs, MPS on the Apple
      M2. The k-means|| init ends with a k-means++ on the host.</li>
      <li>FAISS Lloyd (GPU): <code>faiss.Kmeans</code> 1.16, Lloyd with a random
      init and a fixed number of iterations (<code>niter</code>). By default FAISS
      subsamples to 256 samples per cluster: here it trains on all the samples, unless
      the settings show <code>max_points_per_centroid=256</code>. Its training loop
      runs on the host and only the nearest center search runs on the GPU, so its time
      includes the transfers of every iteration.</li>
      <li>FAISS e-means (NumPy): <code>NumpyEmeans</code> of
      <code>faiss.contrib.e_means</code> 1.16, the reference implementation of the
      paper, in NumPy with FAISS's nearest neighbor search.</li>
      <li>FAISS e-means (Triton, FP32): <code>QuantizedEmeans(quantizer=None)</code>,
      e-means on PyTorch tensors with FAISS's Triton kernels, in float32.</li>
      <li>FAISS e-means (BF16): <code>Emeans</code> with its defaults, which turn the
      float32 input into bfloat16 operands (reduced precision).</li>
      <li>cuML KMeans: RAPIDS cuML <code>KMeans</code>. Its <code>"k-means++"</code>
      init is greedy k-means++, and <code>"scalable-k-means++"</code> is k-means||.</li>
    </ul>
    <p>Settings don't mean the same thing everywhere, which is why the hover shows the
    iterations next to the time. <code>tol</code> is relative to the mean feature
    variance in scikit-learn, and an absolute change of the inertia in
    cuML. FAISS Lloyd and e-means have no tolerance: their budget is
    <code>niter</code> or <code>n_epochs</code> (passes over the data, fractional
    allowed). The batch size and step size of e-means come from the paper's scaling
    laws, and <code>batch_law="quality"</code> uses 8x smaller batches.</p>
    <p>20 Newsgroups and RCV1 are text, as sparse CSR matrices with 130k and 47k
    features. Only scikit-learn Lloyd and sklearn-contrib-emeans take sparse input, so
    they are the only ones on these plots. Lloyd gets slow there (~50 s per iteration
    on RCV1 with 1024 clusters on 24 cores), so it only runs up to 30 iterations, and
    not on RCV1 with 1024 clusters.</p>
  </details>
</section>"""


def implementation_label(case: dict) -> str:
    library = case["implementation"]["library"]
    estimator = case["algorithm"]["estimator"]
    params = case["algorithm"]["estimator_params"]
    if library == "sklearn":
        if estimator == "KMeans":
            return f"scikit-learn {params.get('algorithm', 'lloyd').capitalize()}"
        return f"scikit-learn {estimator}"
    if library == "faiss":
        return {
            "Kmeans": f"FAISS Lloyd ({'GPU' if params.get('gpu') else 'CPU'})",
            "NumpyEmeans": "FAISS e-means (NumPy)",
            "FastNumpyEmeans": "FAISS e-means (Cython kernel)",
            "QuantizedEmeans": "FAISS e-means (Triton, FP32)",
            "Emeans": "FAISS e-means (BF16)",
        }[estimator]
    if library == "sklearn_emeans":
        if case["implementation"].get("data_library") == "torch":
            return "sklearn-contrib-emeans (PyTorch)"
        return "sklearn-contrib-emeans"
    if library == "emeans":
        return "emeans (Rust)"
    if library == "cuml":
        return "cuML KMeans"
    if library == "kmeans_lloyd":
        return "CuPy k-means (fused kernels)" if params.get("fused") else "CuPy k-means"
    return f"{library} {estimator}"


# sklearn-contrib-emeans is plotted with MKL on Intel CPUs and OpenBLAS on the others.
# The other build's runs still count for the best known inertia.
MKL_BUILD = "sklearn-contrib-emeans-mkl"


@cache
def _intel_cpu(hardware_hash: str) -> bool:
    return "Intel" in read_env("hardware", hardware_hash)["CPU"]["name"]


def _plotted_build(record: BenchmarkRecord) -> bool:
    if implementation_label(record.case) != "sklearn-contrib-emeans":
        return True
    is_mkl = software_build_name(record.software_hash) == MKL_BUILD
    return is_mkl == _intel_cpu(record.hardware_hash)


def _threads(record: BenchmarkRecord) -> str | None:
    return (record.case.get("env") or {}).get("OMP_NUM_THREADS")


def _series_key(record: BenchmarkRecord) -> tuple:
    return (
        implementation_label(record.case),
        record.hardware_hash,
        software_build_name(record.software_hash),
        _threads(record),
    )


def read_records() -> list[BenchmarkRecord]:
    """The latest successful record per case, build and machine."""
    latest = {}
    for record in read_benchmark_records():
        if not record.runs or not matches_source_configs(record.case, SOURCE_CONFIGS):
            continue
        # A case of a subset config (kmeans_pareto_fast.py) is the same case.
        key = (
            record.hardware_hash,
            software_build_name(record.software_hash),
            stable_json(without_keys(record.case, excluded_names={"source_config"})),
        )
        if key not in latest or record.timestamp_recorded > latest[key].timestamp_recorded:
            latest[key] = record
    return list(latest.values())


def _plot_key(record: BenchmarkRecord) -> tuple[str, int]:
    return record.case["data"]["dataset"], record.case["algorithm"]["estimator_params"]["n_clusters"]


def _inertias(record: BenchmarkRecord) -> list[float]:
    return [run["metrics"]["fit"]["inertia_per_sample"] for run in record.runs]


def _median_of(record: BenchmarkRecord, section: str, name: str):
    values = [
        run.get(section, {}).get(name)
        for run in record.runs
        if run.get(section, {}).get(name) is not None
    ]
    return statistics.median(values) if values else None


def _settings(case: dict) -> str:
    params = case["algorithm"]["estimator_params"]
    return " ".join(
        f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"
        for name, value in sorted(params.items())
        if name not in IMPLEMENTATION_PARAMS
    )


def _point(record: BenchmarkRecord, best: float) -> dict:
    times = [run["time_ms"]["fit"] / 1000 for run in record.runs]
    relative = [inertia / best - 1 for inertia in _inertias(record)]
    return {
        "record": record,
        "fit_s": statistics.median(times),
        "fit_s_min": min(times),
        "fit_s_max": max(times),
        "rel": statistics.median(relative),
        "rel_min": min(relative),
        "rel_max": max(relative),
    }


def pareto_front(points: list[dict]) -> list[dict]:
    """The points that no faster point beats on relative inertia."""
    front, best = [], math.inf
    for point in sorted(points, key=lambda p: (p["fit_s"], p["rel"])):
        if point["rel"] < best:
            front.append(point)
            best = point["rel"]
    return front


def _relative(value: float) -> str:
    return "best" if value < TIED_WITH_BEST else f"{value:.2e}"


def _hover(point: dict, name: str) -> str:
    record = point["record"]
    attributes = [run.get("attributes", {}) for run in record.runs]
    lines = [
        f"<b>{escape(name)}</b>",
        escape(_settings(record.case)),
        f"fit time: {point['fit_s']:.3g} s ({point['fit_s_min']:.3g} to {point['fit_s_max']:.3g})",
        f"relative inertia: {_relative(point['rel'])} "
        f"({_relative(point['rel_min'])} to {_relative(point['rel_max'])})",
    ]
    n_iter = [a["n_iter"] for a in attributes if isinstance(a.get("n_iter"), (int, float))]
    epochs = [a["epochs"] for a in attributes if a.get("epochs") is not None]
    if n_iter:
        lines.append(f"iterations (best init): {statistics.median(n_iter):g}")
    if epochs:
        lines.append(f"passes over the data: {statistics.median(epochs):.3g}")
    n_empty = max(run["metrics"]["fit"].get("n_empty_clusters", 0) for run in record.runs)
    if n_empty:
        lines.append(f"empty clusters: up to {n_empty}")
    peak = _median_of(record, "measures", "peak_memory_bytes")
    if peak is not None:
        lines.append(f"peak memory: {peak / 2**20:.0f} MiB")
    clock = _median_of(record, "measures", "gpu_sm_clock_mhz_mean")
    if clock is not None:
        lines.append(f"mean SM clock: {clock:.0f} MHz")
    if record.case["implementation"].get("device") != "cuda" and _threads(record):
        lines.append(f"threads: {_threads(record)}")
    lines.append(f"build: {escape(software_build_name(record.software_hash))}")
    return "<br>".join(lines)


def _series_names(series_keys: list[tuple]) -> dict[tuple, str]:
    """Label, machine, then the build or thread count only where they differ."""
    builds, threads = defaultdict(set), defaultdict(set)
    for label, machine, build, n_threads in series_keys:
        builds[label, machine].add(build)
        threads[label, machine].add(n_threads)
    names = {}
    for key in series_keys:
        label, machine, build, n_threads = key
        name = f"{label} · {_machine_name(machine)}"
        if len(builds[label, machine]) > 1:
            name += f" · {build}"
        if len(threads[label, machine]) > 1:
            name += f" · {n_threads} threads"
        names[key] = name
    return names


def _device(label: str) -> str:
    return IMPLEMENTATIONS.get(label, (None, "CPU"))[1]


def _machine_name(hardware_hash: str) -> str:
    return HARDWARE_NAMES.get(hardware_hash, hardware_hash)


def _plot_html(records: list[BenchmarkRecord], best: float, machines: list[str]) -> str:
    by_series = defaultdict(list)
    for record in records:
        by_series[_series_key(record)].append(_point(record, best))
    values = [v for points in by_series.values() for p in points for v in (p["rel"], p["rel_min"])]

    def clip(value):
        return max(value, BEST_LINE)

    names = _series_names(list(by_series))
    order = list(IMPLEMENTATIONS)
    fig = go.Figure()
    for key in sorted(by_series, key=lambda k: (order.index(k[0]) if k[0] in order else len(order), k)):
        label, machine, _, _ = key
        points = sorted(by_series[key], key=lambda p: p["fit_s"])
        front = pareto_front(points)
        color = IMPLEMENTATIONS.get(label, ("#48566b",))[0]
        symbol = MACHINE_SYMBOLS[machines.index(machine) % len(MACHINE_SYMBOLS)]
        meta = {"implementation": label, "machine": machine}
        visible = label in DEFAULT_IMPLEMENTATIONS and machine in DEFAULT_MACHINES
        fig.add_trace(go.Scatter(
            x=[p["fit_s"] for p in front],
            y=[clip(p["rel"]) for p in front],
            # The markers duplicate the front points of the next trace, so that the
            # legend tells the machines apart.
            mode="lines+markers",
            line={"color": color, "width": 2.5, "shape": "hv"},
            marker={"color": color, "size": 10, "symbol": symbol,
                    "line": {"color": "white", "width": 0.8}},
            name=names[key],
            legendgroup=names[key],
            hoverinfo="skip",
            meta=meta,
            visible=visible,
        ))
        on_front = {id(p) for p in front}
        fig.add_trace(go.Scatter(
            x=[p["fit_s"] for p in points],
            y=[clip(p["rel"]) for p in points],
            error_x={"type": "data", "symmetric": False, "thickness": 1, "width": 0,
                     "array": [p["fit_s_max"] - p["fit_s"] for p in points],
                     "arrayminus": [p["fit_s"] - p["fit_s_min"] for p in points]},
            error_y={"type": "data", "symmetric": False, "thickness": 1, "width": 0,
                     "array": [clip(p["rel_max"]) - clip(p["rel"]) for p in points],
                     "arrayminus": [clip(p["rel"]) - clip(p["rel_min"]) for p in points]},
            mode="markers",
            marker={"color": color, "size": 10,
                    "symbol": symbol,
                    "opacity": [1.0 if id(p) in on_front else 0.3 for p in points],
                    "line": {"color": "white", "width": 0.8}},
            name=names[key],
            legendgroup=names[key],
            showlegend=False,
            hovertext=[_hover(p, names[key]) for p in points],
            hoverinfo="text",
            meta=meta,
            visible=visible,
        ))
    if any(v < BEST_LINE for v in values):
        fig.add_hline(y=BEST_LINE, line={"color": "#48566b", "width": 1, "dash": "dot"})
        # Annotations take log10 coordinates on a log axis, lines don't.
        fig.add_annotation(
            text="best known inertia (within 1e-5)",
            x=0, xref="paper", xanchor="left",
            y=math.log10(BEST_LINE), yref="y", yanchor="bottom",
            showarrow=False,
            font={"size": 11, "color": "#48566b"},
        )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=560,
        margin={"l": 70, "r": 20, "t": 20, "b": 60},
        legend={"font": {"size": 11}},
        xaxis={"type": "log", "title": "fit time (s)"},
        yaxis={"type": "log", "exponentformat": "power", "title": "relative inertia (inertia / best known \u2212 1)"},
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _checkboxes(name: str, legend: str, values: list[tuple[str, str]], checked: set[str]) -> str:
    boxes = "".join(
        f'<label><input type="checkbox"{" checked" if value in checked else ""}'
        f' data-{name}="{escape(value)}" data-slug="{_slug(text)}"> {escape(text)}</label>'
        for value, text in values
    )
    return (
        f'<fieldset class="cross-variants"><legend>{legend}</legend>'
        f'<div class="cross-variants-boxes">{boxes}</div></fieldset>'
    )


FILTER_SCRIPT = """<script>
(() => {
  function checked(name) {
    return new Set(Array.from(document.querySelectorAll(`input[data-${name}]`))
      .filter((box) => box.checked).map((box) => box.dataset[name]));
  }
  function applyFilters() {
    const implementations = checked("implementation");
    const machines = checked("machine");
    document.querySelectorAll(".kmeans-plot").forEach((cell) => {
      const chart = cell.querySelector(".plotly-graph-div");
      const visible = chart.data.map((trace) =>
        implementations.has(trace.meta.implementation) && machines.has(trace.meta.machine));
      cell.hidden = !visible.some(Boolean);
      Plotly.restyle(chart, {visible: visible});
    });
  }
  const GROUPS = ["implementation", "machine"];
  function boxes(name) {
    return Array.from(document.querySelectorAll(`input[data-${name}]`));
  }
  function writeHash() {
    const parts = GROUPS.flatMap((name) => {
      const all = boxes(name);
      const on = all.filter((box) => box.checked);
      return all.every((box) => box.checked === box.defaultChecked)
        ? [] : [`${name}=${on.map((box) => box.dataset.slug).join(",")}`];
    });
    const url = window.location.pathname + window.location.search;
    history.replaceState(null, "", parts.length ? `${url}#${parts.join("&")}` : url);
  }
  function readHash() {
    const params = new URLSearchParams(window.location.hash.slice(1));
    GROUPS.forEach((name) => {
      const on = params.has(name) ? new Set(params.get(name).split(",")) : null;
      boxes(name).forEach((box) => {
        box.checked = on ? on.has(box.dataset.slug) : box.defaultChecked;
      });
    });
    applyFilters();
  }
  document.addEventListener("change", (event) => {
    if (event.target.matches("input[data-implementation], input[data-machine]")) {
      applyFilters();
      writeHash();
    }
  });
  window.addEventListener("hashchange", readHash);
  document.addEventListener("DOMContentLoaded", readHash);
})();
</script>"""


def render(records: list[BenchmarkRecord]) -> str:
    best = defaultdict(lambda: math.inf)
    for record in records:
        best[_plot_key(record)] = min(best[_plot_key(record)], *_inertias(record))
    plotted = [
        r for r in records
        if r.case["metadata"].get("source_config") != REFERENCE_CONFIG
        and implementation_label(r.case) not in HIDDEN_IMPLEMENTATIONS
        and _plotted_build(r)
    ]
    hashes = {r.hardware_hash for r in plotted}
    machines = [h for h in HARDWARE_NAMES if h in hashes] + sorted(hashes - set(HARDWARE_NAMES))
    labels = {implementation_label(r.case) for r in plotted}
    order = list(IMPLEMENTATIONS)
    labels = sorted(labels, key=lambda label: (order.index(label) if label in order else len(order), label))

    by_plot = defaultdict(list)
    for record in plotted:
        by_plot[_plot_key(record)].append(record)
    datasets = list(DATASET_NAMES)
    cells = []
    for dataset, n_clusters in sorted(
        by_plot, key=lambda key: (datasets.index(key[0]) if key[0] in datasets else len(datasets), key)
    ):
        plot_records = by_plot[dataset, n_clusters]
        shape = plot_records[0].runs[0]["data_desc"]["fit"]
        title = f"{DATASET_NAMES.get(dataset, dataset)}, {n_clusters} clusters"
        subtitle = (
            f"{shape['samples']:,} samples x {shape['features']} features, "
            f"best known inertia {best[dataset, n_clusters]:.6g} per sample"
        )
        cells.append(
            f'<section class="plot-cell kmeans-plot"><h3>{escape(title)}</h3>'
            f'<p class="plot-subtitle">{escape(subtitle)}</p>'
            f"{_plot_html(plot_records, best[dataset, n_clusters], machines)}"
            f'<p class="plot-footnote">{PLOT_FOOTNOTE}</p></section>'
        )

    filters = (
        '<section class="panel">'
        + "".join(
            _checkboxes(
                "implementation",
                f"{device} implementations",
                [(label, label) for label in labels if _device(label) == device],
                DEFAULT_IMPLEMENTATIONS,
            )
            for device in ("CPU", "GPU")
            if any(_device(label) == device for label in labels)
        )
        + _checkboxes("machine", "Machines", [(h, _machine_name(h)) for h in machines], DEFAULT_MACHINES)
        + "</section>"
    )
    for name, values in (("implementation", labels), ("machine", [_machine_name(h) for h in machines])):
        slugs = [_slug(value) for value in values]
        assert len(set(slugs)) == len(slugs), f"{name} checkboxes share a URL slug: {slugs}"
    return BASE_TEMPLATE.render(
        title=TITLE,
        rows=[
            ABOUT_HTML,
            DATE_RANGE_TEMPLATE.render(**date_range(records)),
            filters,
            f'<div class="plot-grid">{"".join(cells)}</div>' + FILTER_SCRIPT,
        ],
    )


def generate(output_dir: Path) -> None:
    output = output_dir / OUTPUT
    output.write_text(render(read_records()))
    print(f"Dashboard written to {output}")
