"""Overall HistGradientBoosting fit-time scaling vs thread count, compared
across sklearn builds on the same machine.

Reuses the instrumented HGB records and thread-count/dedup logic from
gen_hgb_scaling.py (see that module's docstring for why), but instead of one
phase-breakdown tab per (hardware, build), plots total fit time as a line per
build directly overlaid on one chart per workload - simpler, and makes
scaling differences between builds comparable at a glance.
"""
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.gen_hgb_scaling import (
    HARDWARE_NAMES,
    _dedup_latest,
    _is_instrumented_hgb,
    _subtitle_html,
    _thread_count,
    _workload_name,
    _workload_size,
)
from dashboards.output import dashboard_output_path
from sklbench.reporting.envs import read_env, software_build_name, summarize_hardware_env
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    render_hardware_tabs,
    scaling_line_plot_html,
    variant_color_map,
)
from sklbench.reporting.matching import BenchmarkRecord, date_range, read_benchmark_records


def _fit_ms(record: BenchmarkRecord) -> float | None:
    values = [
        run["time_ms"]["fit"] for run in record.runs
        if "time_ms" in run and "fit" in run["time_ms"]
    ]
    return mean(values) if values else None


def render_hardware_page(records: list[BenchmarkRecord]) -> str:
    if not records:
        return '<section class="empty">No instrumented HGB results for this hardware.</section>'

    builds = sorted({software_build_name(record.software_hash) for record in records})
    colors = variant_color_map(builds)

    by_workload: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        by_workload.setdefault(_workload_name(record), []).append(record)

    cells = []
    for name in sorted(by_workload, key=lambda n: _workload_size(by_workload[n][0])):
        series: dict[str, list[tuple[int, float]]] = {}
        for record in by_workload[name]:
            threads = _thread_count(record)
            fit_ms = _fit_ms(record)
            if threads is None or fit_ms is None:
                continue
            build = software_build_name(record.software_hash)
            series.setdefault(build, []).append((threads, fit_ms))
        if not series:
            continue
        n_samples, n_features = _workload_size(by_workload[name][0])
        title = f"{name} ({n_samples:,} x {n_features})"
        subtitle_html = _subtitle_html(by_workload[name][0])
        plot = scaling_line_plot_html(series, colors=colors)
        cells.append(
            f'<section class="plot-cell"><h3>{title}</h3>{subtitle_html}{plot}</section>'
        )

    grid = (
        '<section class="plot-grid" '
        'style="grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));">'
        + "".join(cells)
        + "</section>"
    )
    rows = [
        DATE_RANGE_TEMPLATE.render(**date_range(records)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", records[0].hardware_hash))),
        grid,
    ]
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


if __name__ == "__main__":
    records = _dedup_latest(
        [record for record in read_benchmark_records() if _is_instrumented_hgb(record)]
    )
    by_hardware: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        by_hardware.setdefault(record.hardware_hash, []).append(record)

    pages = [
        (HARDWARE_NAMES.get(hardware_hash, hardware_hash), render_hardware_page(hw_records))
        for hardware_hash, hw_records in sorted(by_hardware.items())
    ]

    html = BASE_TEMPLATE.render(
        title="HGB build scalability comparison",
        rows=[render_hardware_tabs(pages)],
    )
    output = dashboard_output_path("hgb_build_scaling.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
