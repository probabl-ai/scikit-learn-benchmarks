"""HistGradientBoosting fit-time phase breakdown across thread counts.

Reads the instrumented HGB records produced by `configs/hgb_scaling.py`
(`sklbench/runners/estimator/wrappers/instrumented_hgb.py` swaps in an HGB
subclass that times binning/tree-growing sub-phases). One small-multiple
stacked bar per workload, x-axis = thread count, segments = phase.

Deliberately reads raw `read_benchmark_records()` instead of
`read_all_results()`: `MethodResult.case`/`full_match_key` strip the `bench`
key, so thread count (which for this config only lives in
`bench.taskset`/`bench.env.OMP_NUM_THREADS`, not in `case`) isn't part of
the de-dup identity - `read_all_results()` collapses every thread-count
variant of a workload down to whichever ran last. `BenchmarkRecord.case`
strips `bench` for the same reason, so thread count is instead read straight
from `bench.env.OMP_NUM_THREADS` in the record's raw JSON (via
`record.record_path`) - the requested thread count, not an affinity-derived
guess, since some configs (e.g. `hgb_scaling_laptop.py`) set
`OMP_NUM_THREADS` without a matching `taskset`.

Excludes `sklearn-dev*` builds - those are one-off scikit-learn git-checkout
builds (a specific commit/PR branch, see CONTRIBUTING.md's
`setup_sklearn_ref.sh` / `run.sh env@owner:ref` workflow) rather than a
stable environment build, so mixing them into this dashboard's per-build
tabs would make a tab mean "whatever PR happened to run last" instead of a
fixed build. See `gen_hgb_dev_scaling.py` for the sklearn-dev-only
counterpart.
"""
from html import escape
import json
from pathlib import Path
import re
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.envs import (
    active_wait_label_suffix,
    has_active_wait,
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
from sklbench.reporting.matching import (
    BenchmarkRecord,
    date_range,
    is_scaling_benchmark,
    read_benchmark_records,
)


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
    """Whether `record` is an instrumented-HGB result from a thread-scaling
    sweep config (`configs/hgb_scaling.py` or alike, e.g.
    `hgb_scaling_proc_bind.py`/`hgb_scaling_force_active_wait.py` - anything
    tagging `metadata.benchmark_type: scaling`), as opposed to some other
    config's HGB result that happens to carry phase timings too, since
    `sklbench.runners.estimator.loading.wrapped_estimators` instruments every
    HistGradientBoosting* estimator unconditionally regardless of which
    config ran it."""
    if not is_scaling_benchmark(record):
        return False
    estimator = record.case.get("algorithm", {}).get("estimator", "")
    if "HistGradientBoosting" not in estimator:
        return False
    return any("binning_time" in (run.get("attributes") or {}) for run in record.runs)


SKLEARN_DEV_PIXI_ENV = "sklearn-dev"
# Matches "sklearn-dev@..." as well as pixi-env variants of it, e.g.
# "sklearn-dev-libomp@..." (see configs/_implementations.py). Re-derived here
# rather than imported from gen_hgb_speedup_breakdown.py, per this codebase's
# convention of each gen_*.py dashboard owning its own such helpers instead
# of importing another dashboard module's internals.
_SKLEARN_DEV_BUILD_RE = re.compile(rf"^{re.escape(SKLEARN_DEV_PIXI_ENV)}-?.*@")


def _is_sklearn_dev_build(record: BenchmarkRecord) -> bool:
    build_name = software_build_name(record.software_hash)
    return build_name == SKLEARN_DEV_PIXI_ENV or bool(_SKLEARN_DEV_BUILD_RE.match(build_name))


def _raw_bench_env(record: BenchmarkRecord) -> dict:
    if record.record_path is None:
        return {}
    raw_case = json.loads(record.record_path.read_text()).get("case", {})
    return raw_case.get("bench", {}).get("env", {}) or {}


def _thread_count(record: BenchmarkRecord) -> int | None:
    threads = _raw_bench_env(record).get("OMP_NUM_THREADS")
    return int(threads) if threads is not None else None


def _tree_n_threads(record: BenchmarkRecord) -> int | None:
    """The actual OpenMP thread count used to grow the trees (excluding
    binning) - see `instrumented_hgb.py`. Absent on records captured before
    that attribute existed, or on builds without thread-count tuning, in
    which case it's just `_thread_count(record)`."""
    return next(
        (
            run["attributes"]["tree_n_threads"]
            for run in record.runs
            if "tree_n_threads" in (run.get("attributes") or {})
        ),
        None,
    )


def _has_active_wait(record: BenchmarkRecord) -> bool:
    """Whether idle worker threads busy-spin instead of sleeping between
    parallel regions for this record's run - see
    `sklbench.reporting.envs.has_active_wait` for the family-aware
    (GOMP_SPINCOUNT/KMP_BLOCKTIME) override-or-ambient logic. Some runs of
    the same hardware/software combo toggle the override (e.g.
    `hgb_scaling_laptop.py` disabling it), so this has to be part of the
    tab/dedup identity below - otherwise a with/without-override pair for
    the same workload and thread count would collide as if they were reruns
    of each other."""
    return has_active_wait(_raw_bench_env(record), record.software_hash)


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
    fit_ms_repeats = [run["time_ms"]["fit"] for run in runs]
    fit_ms = median(fit_ms_repeats)
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
        # Raw per-repeat fit times (ms), for callers that need to gauge
        # measurement noise around `total_ms` (e.g. gen_hgb_speedup_breakdown.py)
        # rather than just the de-noised median point estimate.
        "total_ms_repeats": fit_ms_repeats,
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
# config and not informative per-plot. learning_rate/l2_regularization are
# tuning knobs that don't affect fit time, so they're noise in a time-scaling
# breakdown.
_SUBTITLE_HPARAMS_EXCLUDED = {
    "max_iter",
    "max_leaf_nodes",
    "max_bins",
    "early_stopping",
    "learning_rate",
    "l2_regularization",
}


def _workload_subtitle(record: BenchmarkRecord) -> str:
    """One small-print line of the hyperparameters/data-shape details that
    most affect fit time but aren't captured by the workload name/shape
    alone - e.g. "L" vs "L-leaf255" in hgb_scaling.py's WORKLOADS only differ
    by `max_leaf_nodes`/`max_iter`, not by name. `n_iter` is the actual
    number of boosting iterations run (from the fitted estimator's
    `n_iter_`), not the `max_iter` param - the two only diverge when
    `early_stopping` is on, but reading the real value avoids being wrong in
    that case. The actual tree-growing thread count (`tree_n_threads`, which
    can be lower than `OMP_NUM_THREADS` on branches that size it down for
    small workloads) varies per thread-count point within a workload, so
    it's shown in the x-axis tick labels instead (see `render_env_page`),
    not here."""
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
    axis_note = (
        '<div class="plot-subtitle">x-axis: requested threads (OMP_NUM_THREADS)'
        " - in parens, the actual thread count used to grow trees"
        " (absent on records without that instrumentation)</div>"
    )
    return f'<div class="phase-legend">{items}</div>{axis_note}'


def _env_summary_rows(records: list[BenchmarkRecord]) -> list[str]:
    hardware_hash = records[0].hardware_hash
    software_hash = records[0].software_hash
    hardware_summary = summarize_hardware_env(read_env("hardware", hardware_hash))
    software_summary = summarize_software_env(
        read_env("software", software_hash),
        records[0].implementation,
        software_hash=software_hash,
        case_env=_raw_bench_env(records[0]),
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
            tree_n_threads = _tree_n_threads(record)
            x_label = f"{threads} ({tree_n_threads})" if tree_n_threads is not None else str(threads)
            points.append(
                {
                    "x": threads,
                    "x_label": x_label,
                    "phases": breakdown,
                    "total_ms": breakdown["total_ms"],
                }
            )
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
            _has_active_wait(record),
        )
        current = latest.get(key)
        if current is None or record.timestamp_recorded > current.timestamp_recorded:
            latest[key] = record
    return list(latest.values())


def _env_key(record: BenchmarkRecord) -> tuple[str, str, bool]:
    return (record.hardware_hash, record.software_hash, _has_active_wait(record))


def _env_label(hardware_hash: str, software_hash: str, active_wait: bool) -> str:
    hardware_label = HARDWARE_NAMES.get(hardware_hash, hardware_hash)
    return (
        f"{hardware_label} — {software_build_name(software_hash)}"
        f"{active_wait_label_suffix(active_wait)}"
    )


if __name__ == "__main__":
    records = _dedup_latest(
        [
            record
            for record in read_benchmark_records()
            if _is_instrumented_hgb(record) and not _is_sklearn_dev_build(record)
        ]
    )
    by_env: dict[tuple[str, str, bool], list[BenchmarkRecord]] = {}
    for record in records:
        by_env.setdefault(_env_key(record), []).append(record)

    # Tabs are one per (hardware, software build, active-wait) combo - each
    # build (e.g. different OpenMP runtime) gets its own tab rather than
    # nesting tabs inside tabs, so the impact of a build swap is a tab click
    # away without doubling up the hardware-tabs template's page-global JS on
    # one page. Active-wait (GOMP_SPINCOUNT) is included too since some runs
    # of the same build toggle it (see `_has_active_wait`).
    pages = [
        (_env_label(hardware_hash, software_hash, active_wait), render_env_page(env_records))
        for (hardware_hash, software_hash, active_wait), env_records in sorted(
            by_env.items(),
            key=lambda item: _env_label(*item[0]),
        )
    ]

    html = BASE_TEMPLATE.render(
        title="HGB fit-time breakdown (thread scalability)",
        rows=[render_hardware_tabs(pages)],
    )
    output = dashboard_output_path("hgb_scaling.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
