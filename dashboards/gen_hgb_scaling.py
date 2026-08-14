"""HistGradientBoosting fit-time phase breakdown across thread counts.

Reads the instrumented HGB records produced by `configs/hgb_scaling.py`
(`sklbench/runners/estimator/wrappers/instrumented_hgb.py` swaps in an HGB
subclass that times binning/tree-growing sub-phases). One small-multiple
stacked bar per workload, x-axis = thread count, segments = phase.

Deliberately reads raw `read_benchmark_records()` instead of
`read_all_results()`: `MethodResult.case`/`full_match_key` strip the `bench`
key, so thread count (which for this config only lives in
`bench.taskset`/`bench.env.OPENMP_NUM_THREADS`, not in `case`) isn't part of
the de-dup identity - `read_all_results()` collapses every thread-count
variant of a workload down to whichever ran last. Thread count is instead
read from each run's `profiling_metrics.fit.n_detected_physical_cores`,
which reflects the taskset-restricted affinity the run actually saw.
"""
from html import escape
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.envs import (
    read_env,
    software_build_name,
    summarize_hardware_env,
    summarize_software_env,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    PLOTLY_DEFAULT_COLORS,
    SOFTWARE_TEMPLATE,
    phase_breakdown_plot_html,
    render_hardware_tabs,
)
from sklbench.reporting.matching import BenchmarkRecord, date_range, read_benchmark_records


HARDWARE_NAMES = {
    "534824": "Intel GNR",  # TODO: re-rerun
    "3b5e61": "Laptop",
}

# Bottom-to-top stack order: phases with a roughly thread-count-independent
# cost first, so their band stays a constant height and the phases that
# actually shrink as threads increase are easy to read off the top.
# `other` absorbs grower_init/make_predictor (both small, per-tree/once-per-fit
# bookkeeping, not worth their own segment) plus any unmeasured residual;
# binning splits into its two sub-timers - computing the bin thresholds
# (fit) vs. applying them to the training data (transform) - since those are
# the two steps that could plausibly parallelize differently.
PHASE_ORDER = [
    "bin_fit_time",
    "bin_transform_time",
    "other",
    "apply_split_time",
    "find_split_time",
    "hist_time",
]
PHASE_LABELS = {
    "bin_fit_time": "bin fit",
    "bin_transform_time": "bin transform",
    "other": "other / unmeasured",
    "apply_split_time": "apply split",
    "find_split_time": "find split",
    "hist_time": "compute hist",
}
PHASE_COLORS = dict(zip(PHASE_ORDER, PLOTLY_DEFAULT_COLORS))

# Raw attribute names (seconds) summed from grow_time's/binning_time's
# sub-phases plus the outer fit-time residual - see instrumented_hgb.py for
# what each measures.
_SECONDS_ATTRIBUTES = [
    "binning_time",
    "bin_fit_time",
    "bin_transform_time",
    "grower_init_time",
    "grow_time",
    "make_predictor_time",
    "hist_time",
    "find_split_time",
    "apply_split_time",
]


def _is_instrumented_hgb(record: BenchmarkRecord) -> bool:
    estimator = record.case.get("algorithm", {}).get("estimator", "")
    if "HistGradientBoosting" not in estimator:
        return False
    return any("binning_time" in (run.get("attributes") or {}) for run in record.runs)


def _thread_count(record: BenchmarkRecord) -> int | None:
    for run in record.runs:
        cores = (run.get("profiling_metrics") or {}).get("fit", {}).get(
            "n_detected_physical_cores"
        )
        if cores:
            return int(cores)
    return None


def _fit_within_budget(values: dict[str, float], budget: float) -> tuple[dict[str, float], float]:
    """Scale `values` down (never up) so they sum to at most `budget`,
    preserving their relative proportions, and return the leftover
    (`budget - sum`, clamped to >= 0). Guarantees `sum(scaled) + leftover ==
    budget` exactly either way - used so a stack of measured sub-phases
    always sums to the wall-clock time of the phase they're part of, even
    when the sub-timers overcount it (see `_phase_breakdown_ms`)."""
    total = sum(values.values())
    if total > budget and total > 0:
        scale = budget / total
        return {name: value * scale for name, value in values.items()}, 0.0
    return values, max(budget - total, 0.0)


def _phase_breakdown_ms(record: BenchmarkRecord) -> dict | None:
    """Median phase timings (ms) across a record's repeats, additive to the
    median fit time. Median (not mean) so a single stalled repeat - this
    hardware occasionally sees one repeat balloon 100-1000x, e.g. from
    thermal/scheduler noise on the host, not the code under test - doesn't
    dominate the aggregate.

    `hist_time`/`find_split_time`/`apply_split_time` come from sklearn's own
    per-node timers (`TreeGrower.total_compute_hist_time` & co in
    grower.py), which start accumulating in `TreeGrower.__init__` itself (the
    root node's histogram/split, in `_initialize_root`) - a wall-clock window
    our wrapper already counts separately as `grower_init_time` - and keep
    accumulating through `grow()`. So they're sub-phases of
    `grower_init_time + grow_time` combined, not of `grow_time` alone;
    comparing them to `grow_time` alone double-counts the root node and can
    make them sum to more than `grow_time` by itself. `binning_time` is
    likewise split into its own measured sub-phases (bin_fit/bin_transform).
    `other` collects whatever's left of each budget, the small
    make_predictor_time phase, and any fit-time left unexplained by every
    instrumented phase (estimator setup/validation outside the timed
    block)."""
    runs = [run for run in record.runs if "binning_time" in (run.get("attributes") or {})]
    if not runs:
        return None
    fit_ms = median(run["time_ms"]["fit"] for run in runs)
    seconds = {
        name: median(run["attributes"][name] for run in runs)
        for name in _SECONDS_ATTRIBUTES
    }
    ms = {name: value * 1000 for name, value in seconds.items()}

    tree_budget_ms = ms["grower_init_time"] + ms["grow_time"]
    grow_parts, grow_other_ms = _fit_within_budget(
        {
            "hist_time": ms["hist_time"],
            "find_split_time": ms["find_split_time"],
            "apply_split_time": ms["apply_split_time"],
        },
        tree_budget_ms,
    )
    binning_parts, binning_other_ms = _fit_within_budget(
        {"bin_fit_time": ms["bin_fit_time"], "bin_transform_time": ms["bin_transform_time"]},
        ms["binning_time"],
    )
    measured_ms = ms["binning_time"] + tree_budget_ms + ms["make_predictor_time"]
    other_ms = (
        max(fit_ms - measured_ms, 0.0)
        + binning_other_ms
        + grow_other_ms
        + ms["make_predictor_time"]
    )
    return {
        "bin_fit_time": binning_parts["bin_fit_time"],
        "bin_transform_time": binning_parts["bin_transform_time"],
        "other": other_ms,
        "apply_split_time": grow_parts["apply_split_time"],
        "find_split_time": grow_parts["find_split_time"],
        "hist_time": grow_parts["hist_time"],
        "total_ms": fit_ms,
    }


def _workload_name(record: BenchmarkRecord) -> str:
    data = record.case.get("data", {})
    return (
        data.get("id")
        or data.get("dataset")
        or record.case.get("metadata", {}).get("name")
        or "unknown"
    )


def _workload_size(record: BenchmarkRecord) -> tuple[int, int]:
    generation_kwargs = record.case.get("data", {}).get("generation_kwargs")
    if generation_kwargs:
        return (
            generation_kwargs.get("n_samples", 0),
            generation_kwargs.get("n_features", 0),
        )
    # Real datasets (`data.dataset` instead of `generation_kwargs`) don't
    # carry their shape in the case - read it from a run's recorded
    # `data_desc` instead.
    for run in record.runs:
        fit_desc = (run.get("data_desc") or {}).get("fit", {})
        samples, features = fit_desc.get("samples"), fit_desc.get("features")
        if samples and features:
            return (samples, features)
    return (0, 0)


# Shown separately (n_iter, max_leaf_nodes, categorical count) rather than
# folded into the list below, so they're excluded here to avoid duplicating
# them; max_bins/early_stopping are fixed across every workload in this
# config and not informative per-plot.
_SUBTITLE_HPARAMS_EXCLUDED = {"max_iter", "max_leaf_nodes", "max_bins", "early_stopping"}


def _workload_subtitle(record: BenchmarkRecord) -> str:
    """One small-print line of the hyperparameters/data-shape details that
    most affect fit time but aren't captured by the workload name/shape
    alone - e.g. "L" vs "L-leaf255" in hgb_scaling.py's WORKLOADS only differ
    by `max_leaf_nodes`/`max_iter`, not by name. `n_iter` is the actual
    number of boosting iterations run (from the fitted estimator's
    `n_iter_`), not the `max_iter` param - the two only diverge when
    `early_stopping` is on, but reading the real value avoids being wrong in
    that case."""
    estimator_params = record.case.get("algorithm", {}).get("estimator_params", {})
    parts = []
    n_iter = next(
        (
            run["attributes"]["n_iter"]
            for run in record.runs
            if "n_iter" in (run.get("attributes") or {})
        ),
        None,
    )
    if n_iter is not None:
        parts.append(f"n_iter={n_iter}")
    max_leaf_nodes = estimator_params.get("max_leaf_nodes")
    if max_leaf_nodes is not None:
        parts.append(f"max_leaf_nodes={max_leaf_nodes}")
    n_categorical = next(
        (
            run["data_desc"]["fit"]["n_categorical_features"]
            for run in record.runs
            if "n_categorical_features" in (run.get("data_desc") or {}).get("fit", {})
        ),
        None,
    )
    if n_categorical:
        parts.append(f"{n_categorical} categorical")
    parts += [
        f"{key}={value}"
        for key, value in sorted(estimator_params.items())
        if key not in _SUBTITLE_HPARAMS_EXCLUDED
    ]
    return ", ".join(parts)


def _subtitle_html(record: BenchmarkRecord) -> str:
    subtitle = _workload_subtitle(record)
    return f'<div class="plot-subtitle">{escape(subtitle)}</div>' if subtitle else ""


def _legend_html() -> str:
    items = "".join(
        f'<span><span class="legend-swatch" style="background:{PHASE_COLORS[phase]}"></span>'
        f"{PHASE_LABELS[phase]}</span>"
        for phase in PHASE_ORDER
    )
    return f'<div class="phase-legend">{items}</div>'


def _env_summary_rows(records: list[BenchmarkRecord]) -> list[str]:
    hardware_hash = records[0].hardware_hash
    software_hash = records[0].software_hash
    hardware_summary = summarize_hardware_env(read_env("hardware", hardware_hash))
    software_summary = summarize_software_env(
        read_env("software", software_hash),
        records[0].implementation,
        software_hash=software_hash,
    )
    return [
        HARDWARE_TEMPLATE.render(hardware_summary),
        SOFTWARE_TEMPLATE.render(**software_summary),
    ]


def render_env_page(records: list[BenchmarkRecord]) -> str:
    by_workload: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        by_workload.setdefault(_workload_name(record), []).append(record)
    if not by_workload:
        return '<section class="empty">No instrumented HGB results for this hardware.</section>'

    cells = []
    for name in sorted(by_workload, key=lambda n: _workload_size(by_workload[n][0])):
        points = []
        for record in by_workload[name]:
            breakdown = _phase_breakdown_ms(record)
            threads = _thread_count(record)
            if breakdown is None or threads is None:
                continue
            points.append({"x": threads, "phases": breakdown, "total_ms": breakdown["total_ms"]})
        if not points:
            continue
        n_samples, n_features = _workload_size(by_workload[name][0])
        title = f"{name} ({n_samples:,} x {n_features})"
        subtitle_html = _subtitle_html(by_workload[name][0])
        plot = phase_breakdown_plot_html(
            points,
            phase_order=PHASE_ORDER,
            phase_colors=PHASE_COLORS,
            phase_labels=PHASE_LABELS,
        )
        cells.append(
            f'<section class="plot-cell"><h3>{title}</h3>{subtitle_html}{plot}</section>'
        )

    grid = (
        '<section class="plot-grid phase-breakdown" '
        'style="grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));">'
        + "".join(cells)
        + "</section>"
    )
    rows = [
        DATE_RANGE_TEMPLATE.render(**date_range(records)),
        *_env_summary_rows(records),
        _legend_html() + grid,
    ]
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


def _dedup_latest(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    """Keep only the latest-timestamp record per (hardware, software,
    workload, thread count). `read_benchmark_records()` returns every
    historical run (unlike `read_all_results()`, which this script otherwise
    deliberately bypasses - see module docstring), so a rerun of the same
    scaling sweep leaves two records at the same thread count for the same
    workload/env; without this, both would be plotted as separate bars at
    the same x position."""
    latest: dict[tuple, BenchmarkRecord] = {}
    for record in records:
        key = (
            record.hardware_hash,
            record.software_hash,
            _workload_name(record),
            _thread_count(record),
        )
        current = latest.get(key)
        if current is None or record.timestamp_recorded > current.timestamp_recorded:
            latest[key] = record
    return list(latest.values())


def _env_key(record: BenchmarkRecord) -> tuple[str, str]:
    return (record.hardware_hash, record.software_hash)


def _env_label(hardware_hash: str, software_hash: str) -> str:
    hardware_label = HARDWARE_NAMES.get(hardware_hash, hardware_hash)
    return f"{hardware_label} — {software_build_name(software_hash)}"


if __name__ == "__main__":
    records = _dedup_latest(
        [record for record in read_benchmark_records() if _is_instrumented_hgb(record)]
    )
    by_env: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records:
        by_env.setdefault(_env_key(record), []).append(record)

    # Tabs are one per (hardware, software build) combo - each build (e.g.
    # different OpenMP runtime) gets its own tab rather than nesting tabs
    # inside tabs, so the impact of a build swap is a tab click away without
    # doubling up the hardware-tabs template's page-global JS on one page.
    pages = [
        (_env_label(hardware_hash, software_hash), render_env_page(env_records))
        for (hardware_hash, software_hash), env_records in sorted(
            by_env.items(),
            key=lambda item: _env_label(*item[0]),
        )
    ]

    html = BASE_TEMPLATE.render(
        title="HGB fit-time breakdown (thread scaling)",
        rows=[render_hardware_tabs(pages)],
    )
    output = dashboard_output_path("hgb_scaling.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
