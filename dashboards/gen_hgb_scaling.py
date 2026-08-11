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
from pathlib import Path
from statistics import mean
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
    "534824": "Intel GNR 172 CPU cores",
    "3b5e61": "Intel laptop with B390 GPU",
}

# Bottom-to-top stack order: phases with a roughly thread-count-independent
# cost first, so their band stays a constant height and the phases that
# actually shrink as threads increase are easy to read off the top.
PHASE_ORDER = [
    "binning_time",
    "other",
    "make_predictor_time",
    "grower_init_time",
    "apply_split_time",
    "find_split_time",
    "hist_time",
]
PHASE_LABELS = {
    "binning_time": "binning",
    "other": "other / unmeasured",
    "make_predictor_time": "make predictor",
    "grower_init_time": "grower init",
    "apply_split_time": "apply split",
    "find_split_time": "find split",
    "hist_time": "compute hist",
}
PHASE_COLORS = dict(zip(PHASE_ORDER, PLOTLY_DEFAULT_COLORS))

# Raw attribute names (seconds) summed from grow_time's sub-phases plus the
# outer fit-time residual - see instrumented_hgb.py for what each measures.
_SECONDS_ATTRIBUTES = [
    "binning_time",
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


def _phase_breakdown_ms(record: BenchmarkRecord) -> dict | None:
    """Mean phase timings (ms) across a record's repeats, additive to the
    mean fit time. `grow_time` isn't itself a segment - it's split into its
    own measured sub-phases (hist/find_split/apply_split) plus whatever grow()
    overhead those don't cover; `other` catches fit-time not explained by any
    instrumented phase (estimator setup/validation outside the timed block)."""
    runs = [run for run in record.runs if "binning_time" in (run.get("attributes") or {})]
    if not runs:
        return None
    fit_ms = mean(run["time_ms"]["fit"] for run in runs)
    seconds = {
        name: mean(run["attributes"][name] for run in runs)
        for name in _SECONDS_ATTRIBUTES
    }
    ms = {name: value * 1000 for name, value in seconds.items()}
    grow_other_ms = max(
        ms["grow_time"] - ms["hist_time"] - ms["find_split_time"] - ms["apply_split_time"],
        0.0,
    )
    measured_ms = ms["binning_time"] + ms["grower_init_time"] + ms["grow_time"] + ms["make_predictor_time"]
    other_ms = max(fit_ms - measured_ms, 0.0) + grow_other_ms
    return {
        "binning_time": ms["binning_time"],
        "other": other_ms,
        "make_predictor_time": ms["make_predictor_time"],
        "grower_init_time": ms["grower_init_time"],
        "apply_split_time": ms["apply_split_time"],
        "find_split_time": ms["find_split_time"],
        "hist_time": ms["hist_time"],
        "total_ms": fit_ms,
    }


def _workload_name(record: BenchmarkRecord) -> str:
    data = record.case.get("data", {})
    return data.get("id") or record.case.get("metadata", {}).get("name", "unknown")


def _workload_size(record: BenchmarkRecord) -> tuple[int, int]:
    generation_kwargs = record.case.get("data", {}).get("generation_kwargs", {})
    return (
        generation_kwargs.get("n_samples", 0),
        generation_kwargs.get("n_features", 0),
    )


def _legend_html() -> str:
    items = "".join(
        f'<span><span class="legend-swatch" style="background:{PHASE_COLORS[phase]}"></span>'
        f"{PHASE_LABELS[phase]}</span>"
        for phase in PHASE_ORDER
    )
    return f'<div class="phase-legend">{items}</div>'


def _env_summary_html(records: list[BenchmarkRecord]) -> str:
    hardware_hash = records[0].hardware_hash
    software_hash = records[0].software_hash
    hardware_summary = summarize_hardware_env(read_env("hardware", hardware_hash))
    software_summary = summarize_software_env(
        read_env("software", software_hash),
        records[0].implementation,
        software_hash=software_hash,
    )
    return HARDWARE_TEMPLATE.render(hardware_summary) + SOFTWARE_TEMPLATE.render(**software_summary)


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
        plot = phase_breakdown_plot_html(
            points,
            phase_order=PHASE_ORDER,
            phase_colors=PHASE_COLORS,
            phase_labels=PHASE_LABELS,
        )
        cells.append(f'<section class="plot-cell"><h3>{title}</h3>{plot}</section>')

    grid = (
        '<section class="plot-grid phase-breakdown" '
        'style="grid-template-columns: repeat(3, minmax(0, 1fr));">'
        + "".join(cells)
        + "</section>"
    )
    header = DATE_RANGE_TEMPLATE.render(**date_range(records)) + _env_summary_html(records)
    return header + _legend_html() + grid


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
