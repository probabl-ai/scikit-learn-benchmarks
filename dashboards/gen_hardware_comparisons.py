"""The same software on different machines: one outer tab per software, one inner
tab per machine, any of which can be picked as the baseline. Also provides what
pairing results across machines takes to `gen_cross_comparison`.

A software label (`software_label`) is hardware-agnostic, so that e.g. PyTorch
on Intel's "xpu" and on Apple's "mps" share a tab, but keeps CPU and GPU apart.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import median

from dashboards import (
    GPU_NAMES, HARDWARE_NAMES, GENERAL_SOURCE_CONFIGS, GENERAL_SOURCE_ENVS,
)
from dashboards import gen_hptuning_scalability as hptuning
from sklbench.reporting.comparison import (
    CONTENT_HTML,
    PREPROCESSING_TOGGLE_HTML,
    baseline_pick_html,
    expanded_note,
    page_rows,
    tree_binning_note,
    write_data_files,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SERIES_COLORS,
    SOFTWARE_TEMPLATE,
)
from sklbench.reporting.html.blocks import _slug_id
from sklbench.reporting.envs import (
    read_env,
    software_build_name,
    summarize_hardware_env,
    summarize_software_env,
)
from sklbench.reporting.matching import (
    read_all_results,
    read_failed_records,
    matches_source_configs,
    BenchmarkRecord,
    MethodResult,
)
from sklbench.reporting.utils import stable_json, without_keys


ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares machines running the same software. Since the
  software is fixed, a <dfn>speed-up</dfn> comes from the hardware.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Pick a software tab. The CPU and GPU runs of a library are separate
    tabs, like <code>sklearn-torch-cpu</code> and <code>sklearn-torch-gpu</code>.
    Below it, each machine has a tab describing the machine and its
    <dfn>software environment</dfn>, since versions can differ between machines.
    To change the <dfn>baseline</dfn>, open another machine tab and click "Pick as
    baseline".</p>
    <p>The plots read like in
    <a href="per_hardware.html">the software/implementations dashboard</a>,
    with one color per machine: each point is one benchmark case, and points
    above the dashed 1x line are faster than the baseline. Marker shapes mean
    the same.</p>
    <p><b>Random forests</b> and <b>extra trees</b> often have a different forest
    size on each machine: their times are rescaled to the same size and their
    metrics aren't compared. The last row compares the time per fit in
    hyper-parameter searches, where many fits run in parallel.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li>The high-end Intel Xeon server mostly helps <b>random forests</b> and <b>extra
      trees</b>. With <b>scikit-learn</b>, their fit is ~2x to 4x faster than on the
      Intel Core Ultra laptop. Their predict is close to parity.</li>
      <li><b>scikit-learn-intelex</b> makes much better use of the server's 172
      cores. With it, <b>random forests</b> and <b>extra trees</b> fit ~10x to 12x
      faster on the server than on the laptop.</li>
      <li>For <b>linear models</b>, the server is often slower than the
      laptop. A single fit doesn't use many cores well, so the laptop's
      faster cores probably win.</li>
      <li><b>HistGradientBoosting</b> results are hard to interpret, because
      of known scalability issues on laptops with <b><dfn>conda-forge</dfn></b> builds and on
      machines with many cores. See
      <a href="https://github.com/scikit-learn/scikit-learn/issues/34764">scikit-learn#34764</a>
      and the <a href="hgb_scaling.html">HistGradientBoosting
      thread-scalability breakdown</a>.</li>
      <li>On the <b><dfn>PyPI</dfn></b> build, the Apple M4 CPU is ~1.5x to 2x slower
      than the Intel Core Ultra laptop.</li>
    </ul>
  </details>
</section>"""


BASE_IMPLEMENTATION = "sklearn"
# "None"/"default" and the plain "cpu" tag all mean "ran on the CPU".
CPU_DEVICES = {None, "default", "cpu"}
TREE_BINNING_NOTE = tree_binning_note("machine")
# Also read by the cross comparison page.
DATA_DIR = "across_machines_data"


def is_gpu_result(result: MethodResult | BenchmarkRecord) -> bool:
    return result.implementation.device not in CPU_DEVICES


def software_label(result: MethodResult | BenchmarkRecord) -> str:
    implementation = result.implementation
    if implementation.library == BASE_IMPLEMENTATION and implementation.data_library is None:
        return software_build_name(result.software_hash)
    device = "gpu" if is_gpu_result(result) else "cpu"
    if implementation.library == BASE_IMPLEMENTATION:
        return f"{implementation.library}-{implementation.data_library}-{device}"
    return f"{implementation.library}-{device}"


def software_sort_key(label: str) -> tuple:
    """CPU softwares first: stock scikit-learn, sklearnex, Array API backends."""
    if label.startswith("sklearn-pypi"):
        rank = 0
    elif label.startswith("sklearn-cf-"):
        rank = 1
    elif label.startswith("sklearnex-"):
        rank = 2
    elif label.startswith("sklearn-"):
        rank = 3
    else:
        rank = 4
    return label.endswith("-gpu"), rank, label


def hardware_label(result: MethodResult | BenchmarkRecord) -> str:
    """A machine goes by its GPU's name for what ran on that GPU."""
    if is_gpu_result(result) and result.hardware_hash in GPU_NAMES:
        return GPU_NAMES[result.hardware_hash]
    return HARDWARE_NAMES[result.hardware_hash]


HARDWARE_ORDER = [*HARDWARE_NAMES.values(), *GPU_NAMES.values()]


def hardware_sort_key(variant: str) -> tuple:
    """Accepts `split_kind_x_variant` columns too."""
    return HARDWARE_ORDER.index(variant.split(" / ")[0]), variant


def hardware_colors(results: list[MethodResult]) -> dict[str, str]:
    """A machine keeps its color across tabs, under its CPU and GPU names."""
    hashes = {result.hardware_hash for result in results}
    colors = {}
    for index, hardware_hash in enumerate(h for h in HARDWARE_NAMES if h in hashes):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        colors[HARDWARE_NAMES[hardware_hash]] = color
        if hardware_hash in GPU_NAMES:
            colors[GPU_NAMES[hardware_hash]] = color
    return colors


# `n_jobs` and RF/ET's `n_estimators` are both derived in `_real_datasets.py`
# from `N_JOBS = floor(0.9 * cpu_count(...))` - the *local* machine's core
# count at config-generation time - so they legitimately differ between two
# machines' runs of what's otherwise the identical case.
# `source_config` is excluded for the same reason as in
# `MethodResult.minimal_match_key`.
_MATCH_EXCLUDED_NAMES = {
    "implementation", "max_bins", "n_jobs", "n_estimators", "source_config",
}

# Same normalization/target as gen_models_scalability.py's
# `NORMALIZED_N_ESTIMATORS`: RF/ET fit and predict time both scale ~linearly
# with forest size, which differs between machines (see above).
NORMALIZED_N_ESTIMATORS = 100


def _n_estimators(result: MethodResult) -> int | None:
    return result.case.get("algorithm", {}).get("estimator_params", {}).get(
        "n_estimators"
    )


def _normalize_tree_result(result: MethodResult) -> MethodResult:
    n_estimators = _n_estimators(result)
    # Preprocessing doesn't depend on the forest size.
    if not n_estimators or result.method == "preprocessing":
        return result
    scale = NORMALIZED_N_ESTIMATORS / n_estimators
    return replace(result, times=[t * scale for t in result.times])


def match_key(result: MethodResult) -> str:
    case = without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES)
    case["method"] = result.method
    return stable_json(case)


def comparison_key(result: MethodResult | BenchmarkRecord) -> str:
    return stable_json(without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES))


# `configs/hptuning.py` sweeps the outer `n_jobs` with one CV split per worker,
# so neither is part of a search's identity: each fit is the same.
_SEARCH_MATCH_EXCLUDED_NAMES = {"implementation", "max_bins", "n_jobs", "n_splits", "source_config"}


def search_match_key(result: MethodResult) -> str:
    return stable_json(without_keys(result.case, excluded_names=_SEARCH_MATCH_EXCLUDED_NAMES))


def _round_floats(value):
    """`np.logspace` grids can differ in the last digit between numpy builds."""
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, dict):
        return {key: _round_floats(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _search_result(record: BenchmarkRecord) -> MethodResult | None:
    """A search as a "fit" result whose times are the mean time per `.fit()`
    call (see `gen_hptuning_scalability._seconds_per_fit`), which divides out
    the `n_jobs`-dependent `n_splits`."""
    n_fits = hptuning._n_fits(record)
    times = [run["duration_s"] * 1000 / n_fits for run in record.runs if "duration_s" in run]
    if not times:
        return None
    n_samples, n_features = hptuning._fit_shape([record])
    if n_samples is None:
        n_samples, n_features = hptuning._dataset_shape(record)
    case = _round_floats(without_keys(record.case, excluded_names={"bench"}))
    case["implementation"] = record.case.get("implementation") or {"library": BASE_IMPLEMENTATION}
    return MethodResult(
        hardware_hash=record.hardware_hash,
        software=software_build_name(record.software_hash),
        software_hash=record.software_hash,
        method="fit",
        timestamp_recorded=record.timestamp_recorded,
        case=case,
        times=times,
        data_desc={"samples": n_samples, "features": n_features},
        metrics={},
        record=record,
    )


def read_search_results() -> list[MethodResult]:
    """One result per (machine, env, search), at the outer `n_jobs` with the
    lowest time per fit: the setting a user would pick on that machine."""
    records = hptuning.read_records()
    best: dict[tuple, MethodResult] = {}
    for record in records:
        result = _search_result(record)
        if result is None:
            continue
        key = (result.hardware_hash, result.software_hash, search_match_key(result))
        if key not in best or median(result.times) < median(best[key].times):
            best[key] = result
    return list(best.values())


SEARCH_NOTE = (
    "Searches from the "
    '<a href="hptuning_scalability.html">hyper-parameter search dashboard</a>. '
    "Time per fit over the whole search, with each machine at its fastest outer "
    "<code><dfn>n_jobs</dfn></code>. Each search fits the same 8 <dfn>candidates</dfn> on "
    "<code>n_jobs</code> <dfn>CV splits</dfn>, so every fit is the same on both machines and "
    "every worker has 8 fits to do."
)


# Page header and index page label.
TITLE = "Hardware comparison"
SOURCE_CONFIGS = GENERAL_SOURCE_CONFIGS
# The conda-forge BLAS/OpenMP build variants are a software comparison
# (gen_builds_comparison.py); across hardware they'd only multiply the lines
# per cell, so sklearn-cf-mkl alone stands in for the conda-forge builds.
SOURCE_ENVS = [
    env
    for env in GENERAL_SOURCE_ENVS
    if not env.startswith("sklearn-cf-") or env == "sklearn-cf-mkl"
]


@lru_cache(maxsize=None)
def _pixi_env_name(software_hash: str) -> str:
    return read_env("software", software_hash)["pixi_environment_name"]


def _is_excluded_env(result: MethodResult | BenchmarkRecord) -> bool:
    name = _pixi_env_name(result.software_hash)
    return name.startswith("sklearn-cf-") and name not in SOURCE_ENVS


def _latest_per_case(results: list[MethodResult]) -> list[MethodResult]:
    """Old runs under a previous env can differ from the new ones by the
    `n_jobs`/`n_estimators` that `match_key` ignores, which would plot a stale
    point next to the fresh one."""
    latest: dict[tuple, MethodResult] = {}
    for result in results:
        key = (
            result.hardware_hash,
            software_label(result),
            match_key(result),
            result.case.get("algorithm", {}).get("estimator_params", {}).get("max_bins"),
        )
        if key not in latest or result.timestamp_recorded > latest[key].timestamp_recorded:
            latest[key] = result
    return list(latest.values())


def read_results() -> tuple[list[MethodResult], list[BenchmarkRecord]]:
    """Results and failed records of the named machines, comparable across
    machines. A failure already superseded by a successful run is dropped."""
    def is_kept(result: MethodResult | BenchmarkRecord) -> bool:
        return (
            result.hardware_hash in HARDWARE_NAMES
            and matches_source_configs(result.case, SOURCE_CONFIGS)
            and not _is_excluded_env(result)
        )

    results = [
        _normalize_tree_result(result)
        for result in _latest_per_case(
            [result for result in read_all_results() if is_kept(result)]
        )
    ]
    latest_success = {}
    for result in results:
        key = (result.hardware_hash, software_label(result), comparison_key(result))
        if key not in latest_success or result.timestamp_recorded > latest_success[key]:
            latest_success[key] = result.timestamp_recorded
    failed_records = []
    for record in read_failed_records():
        if not is_kept(record):
            continue
        key = (record.hardware_hash, software_label(record), comparison_key(record))
        if key not in latest_success or record.timestamp_recorded > latest_success[key]:
            failed_records.append(record)
    return results, failed_records


def read_kept_search_results() -> list[MethodResult]:
    return [
        result for result in read_search_results()
        if result.hardware_hash in HARDWARE_NAMES and not _is_excluded_env(result)
    ]


@dataclass(frozen=True)
class Variant:
    software: str
    hardware_hash: str

    @property
    def key(self) -> str:
        return f"{self.software}__{self.hardware_hash}"


def variant_of(result: MethodResult | BenchmarkRecord) -> Variant:
    return Variant(software_label(result), result.hardware_hash)


def _group_by_variant(results) -> dict[Variant, list]:
    grouped: dict[Variant, list] = {}
    for result in results:
        grouped.setdefault(variant_of(result), []).append(result)
    return grouped


def _env_html(_variant_key: str, result: MethodResult) -> str:
    summary = summarize_software_env(
        read_env("software", result.software_hash),
        result.implementation,
        software_hash=result.software_hash,
    )
    summary["name"] = software_label(result)
    return (
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", result.hardware_hash)))
        + SOFTWARE_TEMPLATE.render(**summary)
    )


@lru_cache(maxsize=None)
def write_variant_data(output_dir: Path) -> dict[Variant, list[MethodResult]]:
    """Shared with the cross comparison page. Returns the results per variant."""
    results, failed_records = read_results()
    results_by_variant = _group_by_variant(results)
    with_keys = lambda grouped: {variant.key: items for variant, items in grouped.items()}
    write_data_files(
        output_dir,
        DATA_DIR,
        with_keys(results_by_variant),
        failed_by_variant=with_keys(_group_by_variant(failed_records)),
        search_by_variant=with_keys(_group_by_variant(read_kept_search_results())),
        case_key=comparison_key,
        search_key=search_match_key,
        env_html=_env_html,
    )
    return results_by_variant


def generate(output_dir: Path) -> None:
    results_by_variant = write_variant_data(output_dir)
    results = [res for variant_results in results_by_variant.values() for res in variant_results]
    colors = hardware_colors(results)
    softwares = sorted({variant.software for variant in results_by_variant}, key=software_sort_key)

    outer_tabs = []
    for software in softwares:
        variants = sorted(
            (
                (hardware_label(variant_results[0]), variant)
                for variant, variant_results in results_by_variant.items()
                if variant.software == software
            ),
            key=lambda item: hardware_sort_key(item[0]),
        )
        if len(variants) < 2:
            continue
        outer_tabs.append({
            "label": software,
            "slug": _slug_id(software),
            "emptyScope": software,
            "variants": [
                {
                    "key": variant.key,
                    "label": label,
                    "plotted": True,
                    "tabColor": colors[label],
                    "pickHtml": baseline_pick_html(label, colors[label]),
                }
                for label, variant in variants
            ],
        })

    config = {
        "page": "baseline-tabs",
        "dataDir": DATA_DIR,
        "outerTabs": outer_tabs,
        "colors": colors,
        "columnTitle": "Hardware",
        "treeNote": expanded_note(TREE_BINNING_NOTE),
        "splitsFromParams": True,
        "labelOrder": HARDWARE_ORDER,
        "searchNote": expanded_note(SEARCH_NOTE),
    }
    html = BASE_TEMPLATE.render(title=TITLE, rows=[
        ABOUT_HTML,
        PREPROCESSING_TOGGLE_HTML,
        CONTENT_HTML,
        *page_rows(config),
    ])

    output = output_dir / "hardware_comparisons.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
