"""Thread-count scalability dashboard for `configs/models_scalability.py`.

One tab per hardware; within a tab, one row of plots per (estimator, dataset)
pair from that config - preceded by a small panel naming the dataset (shape,
class count) and the estimator's fixed hyperparameters, see `_row_detail_html`
- and one column per plain-CPU Pixi environment it sweeps (`sklearn-pypi`,
`sklearn-cf-mkl`, `intel`). Each row is its own single-row grid (rather than
one grid for the whole tab) precisely so that detail panel can sit right
above its own row instead of a shared grid's `details_by_row` trailing
placement (see `assemble_plots_in_grid`). Each cell is a fit-time vs.
core-count line plot. RandomForestClassifier/ExtraTreesClassifier are the
only estimators there with both a `with_siblings=True` and `=False` variant
(see `models_scalability.py`'s `_with_scaling_bench` docstring for why) - both
are drawn as separate lines on the same cell so the SMT-vs-no-SMT gap reads
directly off one plot rather than needing a second row. Their fit times are
also normalized to a fixed forest size (see `NORMALIZED_N_ESTIMATORS`) and
plotted on a log y-axis, since that config scales `n_estimators` with core
count and normalized fit time spans two-plus orders of magnitude across the
core sweep - a linear axis would flatten most of that range into an
unreadable near-zero tail. A dashed "perfect scalability" reference line
(see `_perfect_scaling_reference`) is added to their cells too, so the real
curve's departure from ideal linear scaling reads directly off the plot.

Records from that config are identified by `metadata.n_cores` - a key unique
to `_with_scaling_bench`, not set by any other config - combined with the
(estimator, dataset) pairs it actually generates, rather than by importing
the config module itself (`configs/` isn't on this script's import path, and
no other `dashboards/gen_*.py` imports from it - see e.g.
`configs/_implementations.py` being re-derived instead of imported in
gen_hgb_scaling.py).
"""
from html import escape
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.envs import software_build_name
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    assemble_plots_in_grid,
    render_hardware_tabs,
    scaling_line_plot_html,
)
from sklbench.reporting.matching import MethodResult, date_range, read_all_results


HARDWARE_NAMES = {
    "534824": "Intel GNR",
    "3b5e61": "Intel laptop",
}

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

ENV_ORDER = ["sklearn-pypi", "sklearn-cf-mkl", "intel"]

SIBLINGS_LABELS = {True: "with SMT", False: "without SMT"}
SIBLINGS_COLORS = {"with SMT": "#636EFA", "without SMT": "#EF553B"}
SINGLE_SERIES_LABEL = "fit time"

# `models_scalability.py`'s `_with_scaling_bench` sizes tree ensembles as
# `max(24, cores_count * 8)` so every worker has its own tree to build at
# every swept core count - meaning n_estimators (and so raw fit time) grows
# with cores_count independently of any actual scaling effect. Normalizing
# every tree point to this fixed forest size divides that confound out,
# leaving just the scaling behavior.
NORMALIZED_N_ESTIMATORS = 100


def _is_models_scalability_result(result: MethodResult) -> bool:
    if result.method != "fit":
        return False
    estimator = result.case.get("algorithm", {}).get("estimator")
    dataset = result.case.get("data", {}).get("dataset")
    return "n_cores" in result.case.get(
        "metadata", {}
    ) and (estimator, dataset) in MODELS


def _cores(result: MethodResult) -> int:
    return result.case["metadata"]["n_cores"]


def _with_siblings(result: MethodResult) -> bool:
    return result.case["metadata"].get("with_siblings", True)


def _env(result: MethodResult) -> str:
    return software_build_name(result.software_hash)


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
    n_iter = _n_iter(result)
    return f"n_iter: {n_iter}" if n_iter is not None else ""


def _fit_seconds(result: MethodResult, *, estimator: str) -> float:
    seconds = median(result.times) / 1000
    if estimator not in TREE_ESTIMATORS:
        return seconds
    n_estimators = _n_estimators(result)
    if not n_estimators:
        raise ValueError(f"Tree-based result missing n_estimators: {result.case}")
    return seconds * NORMALIZED_N_ESTIMATORS / n_estimators


def _series_for_cell(results: list[MethodResult], estimator: str) -> dict:
    if estimator not in TREE_ESTIMATORS:
        return {
            SINGLE_SERIES_LABEL: [
                (_cores(r), _fit_seconds(r, estimator=estimator), _hover_extra(r))
                for r in results
            ]
        }
    series = {}
    for with_siblings, label in SIBLINGS_LABELS.items():
        points = [
            (_cores(r), _fit_seconds(r, estimator=estimator), _hover_extra(r))
            for r in results
            if _with_siblings(r) == with_siblings
        ]
        if points:
            series[label] = points
    return series


def _y_title(estimator: str) -> str:
    if estimator in TREE_ESTIMATORS:
        return f"fit time / {NORMALIZED_N_ESTIMATORS} trees (s)"
    return "fit time (s)"


PERFECT_SCALING_LABEL = "perfect scalability"


def _perfect_scaling_reference(series: dict) -> dict[str, list[tuple[float, float]]]:
    """A dashed y = y0 * x0 / x reference line, anchored to the lowest core
    count in `series` (usually 1 core) - the fit time that core count would
    need at every other swept core count for this cell's scaling to be
    perfectly linear. `with SMT` is preferred as the anchor series since it's
    the full sweep (see `models_scalability.py`'s `_with_scaling_bench`
    docstring for why `without SMT` doesn't get high core counts on its own -
    both variants share the same swept core counts either way, so the choice
    only affects the anchor point, not the line's x range)."""
    anchor_series = series.get("with SMT") or next(iter(series.values()), [])
    if not anchor_series:
        return {}
    x0, y0 = min(anchor_series, key=lambda point: point[0])[:2]
    xs = sorted({point[0] for points in series.values() for point in points})
    return {PERFECT_SCALING_LABEL: [(x, y0 * x0 / x) for x in xs]}


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


def _row_detail_html(result: MethodResult, estimator: str) -> str:
    lines = [_dataset_line(result), _params_line(result, estimator)]
    subtitles = "".join(f'<div class="plot-subtitle">{escape(line)}</div>' for line in lines)
    return f'<section class="panel"><h3>{escape(estimator)}</h3>{subtitles}</section>'


def render_hardware_page(results: list[MethodResult], hardware_hash: str) -> str:
    hw_results = [result for result in results if result.hardware_hash == hardware_hash]
    if not hw_results:
        return '<section class="empty">No benchmark results for this hardware.</section>'

    sections = [
        f'<div class="page-row">{DATE_RANGE_TEMPLATE.render(**date_range(hw_results))}</div>'
    ]
    for estimator in MODEL_ORDER:
        estimator_results = [
            result
            for result in hw_results
            if result.case["algorithm"]["estimator"] == estimator
        ]
        if not estimator_results:
            continue

        plots = []
        for env in ENV_ORDER:
            env_results = [result for result in estimator_results if _env(result) == env]
            if not env_results:
                continue
            series = _series_for_cell(env_results, estimator)
            is_tree = estimator in TREE_ESTIMATORS
            plots.append(
                {
                    "model": estimator,
                    "env": env,
                    "point_count": len(env_results),
                    "plot": scaling_line_plot_html(
                        series,
                        colors=SIBLINGS_COLORS,
                        x_title="cores",
                        y_title=_y_title(estimator),
                        y_unit="s",
                        x_log=True,
                        y_log=is_tree,
                        reference_lines=(
                            _perfect_scaling_reference(series) if is_tree else None
                        ),
                    ),
                }
            )
        if not plots:
            continue

        detail_html = _row_detail_html(estimator_results[0], estimator)
        grid = assemble_plots_in_grid(
            plots,
            rows={"model": [estimator]},
            columns={"env": ENV_ORDER},
        )
        sections.append(f'<div class="page-row">{detail_html}{grid}</div>')

    return "".join(sections)


if __name__ == "__main__":
    results = [
        result for result in read_all_results() if _is_models_scalability_result(result)
    ]
    hardware_hashes = sorted(
        {result.hardware_hash for result in results},
        key=lambda hardware_hash: HARDWARE_NAMES.get(hardware_hash, hardware_hash),
    )
    hardware_pages = [
        (
            HARDWARE_NAMES.get(hardware_hash, hardware_hash),
            render_hardware_page(results, hardware_hash),
        )
        for hardware_hash in hardware_hashes
    ]

    html = BASE_TEMPLATE.render(
        title="Model thread-scalability",
        rows=[render_hardware_tabs(hardware_pages)],
    )
    output = dashboard_output_path("models_scalability.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
