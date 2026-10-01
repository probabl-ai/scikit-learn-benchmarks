"""Thread-count scalability dashboard for `configs/models_scalability.py`.

One tab per hardware; within a tab, a software-envs panel (see
`_software_tabs_html`) followed by a grid with one fit-time vs. core-count
plot per (estimator, dataset) pair, each swept environment drawn as its own
colored line (see `ENV_COLORS`), same layout as `gen_hptuning_scalability.py`.

RandomForestClassifier/ExtraTreesClassifier also have a `with_siblings=False`
variant on hardware with SMT cores (see `models_scalability.py`'s
`_with_scaling_bench`). The main grid only shows the default SMT variant; the
comparison gets its own section below it, one plot per (estimator, env) - see
`_smt_section_html`. Their fit times are normalized to a fixed forest
size (see `NORMALIZED_N_ESTIMATORS`), since that config scales `n_estimators`
with core count, and plotted on a log y-axis with a per-environment dashed
perfect-scalability reference (see `_perfect_scaling_reference`).

On hardware with SMT cores, the non-tree plots also have a "half core" point
at `n_cores=0.5`: one logical CPU without its SMT sibling (see
`models_scalability.py`'s `_with_half_core_bench`). It's left out of the tree
plots, where it measures the same thing as the 1-core "no SMT" point.

Records from that config are identified by `metadata.source_config` (see
`SOURCE_CONFIGS` and `sklbench.reporting.matching.matches_source_configs`),
stamped at load time by `sklbench.config.loader.load_cases_from_script` -
not by importing the config module itself (`configs/` isn't on this script's
import path, and no other `dashboards/gen_*.py` imports from it - see e.g.
`configs/_utils/implementations.py` being re-derived instead of imported in
gen_hgb_scalability_breakdown.py). `MODELS` below stays as a presentation
detail (row selection/ordering only).
"""
from html import escape
from pathlib import Path
from statistics import median

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import read_env, software_build_name, summarize_software_env
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    render_hardware_tabs,
    render_software_tabs,
    scaling_line_plot_html,
    variant_color_map,
)
from sklbench.reporting.matching import (
    MethodResult, date_range, matches_source_configs, read_all_results,
)


ABOUT_HTML = """<section class="panel">
  <p>This dashboard shows how the wall-clock time of a single
  <code>.fit()</code> call changes with the number of CPU cores.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Each tab is a machine and each plot an (estimator, dataset) pair,
    with one line per software environment. The x-axis is in log scale. For
    tree models, the y-axis is in log scale too, and each environment has a
    grey dashed perfect-scaling line starting from its first point, so the
    gap to it is the lost efficiency. Look for curves that flatten or go up. On machines
    with hyper-threading, the 0.5 point of the non-tree models is a single
    logical CPU, without its SMT sibling, so the step from 0.5 to 1 shows what
    SMT brings on one core. For RandomForest and ExtraTrees, a section below
    compares runs with and without SMT, one plot per environment.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li><b>RandomForest</b> and <b>ExtraTrees</b> scale best, up to the
      172 cores of the Xeon server. <b>scikit-learn-intelex</b> makes
      RandomForest several times faster, but doesn't scale better. On the
      laptop, the gains get smaller past 4 cores, probably because the other
      cores are slower E-cores.</li>
      <li><b>Ridge</b>, <b>LogisticRegression</b> and <b>KMeans</b> with
      scikit-learn stop scaling after a few cores, and can get slower with
      more cores on the Xeon server.</li>
      <li><b>scikit-learn-intelex</b> scales these three models much further,
      but on the Xeon server, they get slower past 16 to 64 cores.</li>
      <li>With the <b>PyPI</b> build, <b>LogisticRegression</b> gets slower
      as soon as it runs on more than one logical CPU. The <b>OpenBLAS</b>
      shipped with scipy 1.18 wakes
      all its threads for tiny operations in L-BFGS-B. This is fixed in
      scipy 2.0 (see
      <a href="https://github.com/scipy/scipy/pull/26193#issuecomment-5886021070">scipy#26193</a>).</li>
      <li>On the laptop, scikit-learn-intelex LogisticRegression looks much
      faster from 8 cores on, but that's a <b>oneDAL</b> bug: on 4 threads or
      more, L-BFGS stops too early and returns a worse model. See
      <a href="https://github.com/uxlfoundation/oneDAL/issues/3820">oneDAL#3820</a>.</li>
    </ul>
  </details>
</section>"""

SMT_NOTE = (
    "SMT (simultaneous multithreading, aka hyper-threading) gives each "
    "physical core two logical cores. \"SMT\" pins both logical siblings of "
    "each selected physical core (the default used above); \"no SMT\" uses "
    "one logical thread per physical core."
)

# (estimator, dataset) pairs from `MODEL_DATASET_PAIRS` in
# configs/models_scalability.py, in that file's row order.
MODELS = [
    ("Ridge", "year_prediction_msd"),
    ("LogisticRegression", "covtype"),
    ("ExtraTreesClassifier", "susy"),
    ("RandomForestClassifier", "fraud"),
    ("KMeans", "fashion_mnist_784"),
]
MODEL_ORDER = [estimator for estimator, _ in MODELS]
TREE_ESTIMATORS = {"RandomForestClassifier", "ExtraTreesClassifier"}

SOURCE_ENVS = ["sklearn-pypi", "sklearn-cf-mkl", "intel"]
SOURCE_CONFIGS = ["configs/models_scalability.py"]
# The `intel` pixi env is sklearn patched with sklearnex; plots name the library.
ENV_LABELS = {"intel": "sklearnex"}
ENV_ORDER = [ENV_LABELS.get(env, env) for env in SOURCE_ENVS]
ENV_COLORS = variant_color_map(ENV_ORDER)

SMT_LABELS = {True: "SMT", False: "no SMT"}

# `metadata.n_cores` of `models_scalability.py`'s half-core point.
HALF_CORE = 0.5

# `models_scalability.py`'s `_with_scaling_bench` sizes tree ensembles as
# `max(24, cores_count * 8)` so every worker has its own tree to build at
# every swept core count - meaning n_estimators (and so raw fit time) grows
# with cores_count independently of any actual scaling effect. Normalizing
# every tree point to this fixed forest size divides that confound out,
# leaving just the scaling behavior.
NORMALIZED_N_ESTIMATORS = 100


def _is_half_core_tree(result: MethodResult) -> bool:
    return (
        _cores(result) == HALF_CORE
        and result.case["algorithm"]["estimator"] in TREE_ESTIMATORS
    )


def _is_charted_pair(result: MethodResult) -> bool:
    """Whether `result` is one of `MODELS`' (estimator, dataset) pairs -
    a presentation/layout detail (row selection and ordering), no longer the
    "is this mine" check (see `SOURCE_CONFIGS`, checked in `generate`)."""
    if result.method != "fit":
        return False
    estimator = result.case.get("algorithm", {}).get("estimator")
    dataset = result.case.get("data", {}).get("dataset")
    return (estimator, dataset) in MODELS


def _cores(result: MethodResult) -> float:
    return result.case["metadata"]["n_cores"]


def _with_siblings(result: MethodResult) -> bool:
    return result.case["metadata"].get("with_siblings", True)


def _env(result: MethodResult) -> str:
    build = software_build_name(result.software_hash)
    return ENV_LABELS.get(build, build)


def _n_estimators(result: MethodResult) -> int | None:
    return result.case.get("algorithm", {}).get("estimator_params", {}).get(
        "n_estimators"
    )


def _n_iter(result: MethodResult) -> int | None:
    # Only iterative solvers report this (e.g. LogisticRegression's default
    # lbfgs) - Ridge's default cholesky solver doesn't, so `attributes` won't
    # have it there.
    values = result.attributes.get("n_iter")
    return values[0] if values else None


def _hover_extra(result: MethodResult) -> str:
    lines = []
    if _cores(result) == HALF_CORE:
        lines.append("1 logical CPU, no SMT sibling")
    n_iter = _n_iter(result)
    if n_iter is not None:
        lines.append(f"n_iter: {n_iter}")
    return "<br>".join(lines)


def _fit_seconds(result: MethodResult, *, estimator: str) -> float:
    seconds = median(result.times) / 1000
    if estimator not in TREE_ESTIMATORS:
        return seconds
    n_estimators = _n_estimators(result)
    if not n_estimators:
        raise ValueError(f"Tree-based result missing n_estimators: {result.case}")
    return seconds * NORMALIZED_N_ESTIMATORS / n_estimators


def _point(result: MethodResult, estimator: str) -> tuple:
    return (_cores(result), _fit_seconds(result, estimator=estimator), _hover_extra(result))


def _series(groups: dict[str, list[MethodResult]], estimator: str) -> dict:
    return {
        label: [_point(r, estimator) for r in results]
        for label, results in groups.items()
        if results
    }


def _y_title(estimator: str) -> str:
    if estimator in TREE_ESTIMATORS:
        return f"fit time / {NORMALIZED_N_ESTIMATORS} trees (s)"
    return "fit time (s)"


PERFECT_SCALING_LABEL = "perfect scalability"
REFERENCE_Y_FLOOR = 0.7


def _perfect_scaling_reference(
    series: dict, anchor_labels: list[str]
) -> dict[str, list[list[tuple[float, float]]]]:
    """One y = y0 * x0 / x segment per anchor series, starting at its first
    point and cut once it drops below `REFERENCE_Y_FLOOR` times the cell's
    fastest point, so a slow single-core baseline doesn't stretch the
    y-axis."""
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        return {}
    y_floor = REFERENCE_Y_FLOOR * min(point[1] for point in all_points)
    xs = sorted({point[0] for point in all_points})
    segments = []
    for label in anchor_labels:
        if label not in series:
            continue
        x0, y0 = min(series[label], key=lambda point: point[0])[:2]
        segment = [(x, y0 * x0 / x) for x in xs if x >= x0 and y0 * x0 / x >= y_floor]
        x_floor = y0 * x0 / y_floor
        if x0 < x_floor < xs[-1]:
            segment.append((x_floor, y_floor))
        segments.append(segment)
    return {PERFECT_SCALING_LABEL: segments}


def _dataset_line(result: MethodResult) -> str:
    dataset = result.case.get("data", {}).get("dataset", "unknown")
    desc = result.data_desc
    dims = f"{desc['samples']:,} rows x {desc['features']} features"
    if desc.get("n_classes"):
        dims += f", {desc['n_classes']} classes"
    return f"dataset: {dataset} ({dims})"


def _params_line(result: MethodResult, estimator: str) -> str:
    params = dict(result.case.get("algorithm", {}).get("estimator_params", {}))
    if estimator in TREE_ESTIMATORS:
        # Varies with core count (see `NORMALIZED_N_ESTIMATORS`) - showing one
        # arbitrary value here would be misleading, since the plot already
        # normalizes it away.
        params.pop("n_estimators", None)
    if not params:
        return "params: (defaults)"
    formatted = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"params: {formatted}"


def _cell_subtitle_html(result: MethodResult, estimator: str) -> str:
    lines = [_dataset_line(result), _params_line(result, estimator)]
    return "".join(f'<div class="plot-subtitle">{escape(line)}</div>' for line in lines)


def _plot_cell_html(
    title: str,
    subtitle: str,
    series: dict,
    estimator: str,
    *,
    colors: dict[str, str],
    line_dashes: dict[str, str] | None = None,
    reference_anchors: list[str],
) -> str:
    is_tree = estimator in TREE_ESTIMATORS
    plot = scaling_line_plot_html(
        series,
        colors=colors,
        line_dashes=line_dashes,
        x_title="cores",
        y_title=_y_title(estimator),
        y_unit="s",
        x_log=True,
        y_log=is_tree,
        reference_lines=(
            _perfect_scaling_reference(series, reference_anchors) if is_tree else None
        ),
    )
    return f'<section class="plot-cell"><h3>{escape(title)}</h3>{subtitle}{plot}</section>'


def _grid_html(cells: list[str]) -> str:
    return (
        '<section class="plot-grid" '
        'style="grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));">'
        + "".join(cells)
        + "</section>"
    )


def _smt_section_html(hw_results: list[MethodResult]) -> list[str]:
    """Page rows comparing SMT vs. no SMT, one plot per (tree estimator,
    env) - empty on hardware without SMT cores, where no "no SMT" variant is
    generated."""
    if all(_with_siblings(result) for result in hw_results):
        return []
    rows = [
        '<div class="page-row"><section class="panel"><h3>SMT vs. no SMT</h3>'
        f'<div class="plot-subtitle">{escape(SMT_NOTE)}</div></section></div>'
    ]
    for estimator in MODEL_ORDER:
        if estimator not in TREE_ESTIMATORS:
            continue
        cells = []
        for env in ENV_ORDER:
            env_results = [
                result
                for result in hw_results
                if result.case["algorithm"]["estimator"] == estimator and _env(result) == env
            ]
            series = _series(
                {
                    label: [r for r in env_results if _with_siblings(r) == with_siblings]
                    for with_siblings, label in SMT_LABELS.items()
                },
                estimator,
            )
            if len(series) < 2:
                continue
            cells.append(
                _plot_cell_html(
                    f"{estimator} / {env}",
                    _cell_subtitle_html(env_results[0], estimator),
                    series,
                    estimator,
                    colors={label: ENV_COLORS[env] for label in series},
                    line_dashes={SMT_LABELS[False]: "dash"},
                    reference_anchors=[SMT_LABELS[True]],
                )
            )
        if cells:
            rows.append(f'<div class="page-row">{_grid_html(cells)}</div>')
    return rows if len(rows) > 1 else []


def _software_tabs_html(hw_results: list[MethodResult]) -> str:
    """Tabbed `SOFTWARE_TEMPLATE` card per swept environment (pixi env,
    package versions, threadpools, OpenMP - see `summarize_software_env`),
    same pattern as e.g. `gen_softwares_comparison.py`/`gen_builds_comparison.py`."""
    cards = []
    for env in ENV_ORDER:
        env_results = [result for result in hw_results if _env(result) == env]
        if not env_results:
            continue
        result = env_results[0]
        summary = summarize_software_env(
            read_env("software", result.software_hash),
            result.implementation,
            software_hash=result.software_hash,
        )
        summary["name"] = env
        cards.append(SOFTWARE_TEMPLATE.render(**summary))
    return render_software_tabs(cards)


def render_hardware_page(results: list[MethodResult], hardware_hash: str) -> str | None:
    hw_results = [result for result in results if result.hardware_hash == hardware_hash]
    if not hw_results:
        return None

    sections = [
        f'<div class="page-row">{DATE_RANGE_TEMPLATE.render(**date_range(hw_results))}</div>',
        f'<div class="page-row">{_software_tabs_html(hw_results)}</div>',
    ]
    cells = []
    for estimator in MODEL_ORDER:
        estimator_results = [
            result
            for result in hw_results
            if result.case["algorithm"]["estimator"] == estimator and _with_siblings(result)
        ]
        series = _series(
            {env: [r for r in estimator_results if _env(r) == env] for env in ENV_ORDER},
            estimator,
        )
        if not series:
            continue
        cells.append(
            _plot_cell_html(
                estimator,
                _cell_subtitle_html(estimator_results[0], estimator),
                series,
                estimator,
                colors=ENV_COLORS,
                reference_anchors=ENV_ORDER,
            )
        )
    if cells:
        sections.append(f'<div class="page-row">{_grid_html(cells)}</div>')
    sections.extend(_smt_section_html(hw_results))

    return "".join(sections)


def generate(output_dir: Path) -> None:
    results = [
        result for result in read_all_results()
        if matches_source_configs(result.case, SOURCE_CONFIGS)
        and _is_charted_pair(result)
        and not _is_half_core_tree(result)
    ]
    hardware_hashes = sorted(
        {result.hardware_hash for result in results},
        key=lambda hardware_hash: (
            list(HARDWARE_NAMES).index(hardware_hash)
            if hardware_hash in HARDWARE_NAMES
            else len(HARDWARE_NAMES),
            HARDWARE_NAMES.get(hardware_hash, hardware_hash),
        ),
    )
    hardware_pages = [
        (
            HARDWARE_NAMES.get(hardware_hash, hardware_hash),
            render_hardware_page(results, hardware_hash),
        )
        for hardware_hash in hardware_hashes
    ]

    html = BASE_TEMPLATE.render(
        title="Models core-count scalability",
        rows=[ABOUT_HTML, render_hardware_tabs(hardware_pages)],
    )
    output = output_dir / "models_scalability.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
