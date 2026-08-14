"""Per-phase HistGradientBoosting speed-up, comparing one or more sklearn
branches against `main` on the same machine.

Reuses the instrumented HGB records, thread-count/dedup logic and phase
breakdown from gen_hgb_scaling.py (see that module's docstring for why raw
`read_benchmark_records()` is read instead of `read_all_results()`).

One chart per thread count. Within a chart, x-axis = phase (categorical),
each phase split into one column per variant build (so more than a single
baseline-vs-one-variant comparison reads at a glance, color also encodes
variant); y-axis = each phase's speed-up normalized by the variant's total
fit time, `(baseline_phase_ms - variant_phase_ms) / variant_total_ms` - this
normalization makes a variant's phase contributions additive: they sum to
that variant's overall (baseline_total - variant_total) / variant_total.
Multiple points can land on the same (phase, variant) column - one per
dataset/workload - same convention as gen_per_hardware.py's speedup_plot_html.

Restricted to "sklearn-dev" builds - see CONTRIBUTING.md's
`setup_sklearn_ref.sh` / `run.sh env@owner:ref` workflow, which makes
`software_build_name` return e.g. `sklearn-dev@scikit-learn:main` vs
`sklearn-dev@cakedev0:my-branch`. This is a branch-vs-branch comparison, not
a build-variant one (unlike gen_builds_comparison.py, which compares e.g.
`sklearn-mkl`/`sklearn-openblas-openmp` against plain `sklearn`), so builds
outside `sklearn-dev` (different BLAS/OpenMP runtime, not different sklearn
source) are excluded rather than folded in as more variants. The baseline is
picked dynamically as whichever `sklearn-dev` build's ref is `main` (the
`<owner>` half varies depending on which fork/remote the comparison run
used), rather than a fixed constant.
"""
from collections import defaultdict
from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.gen_hgb_scaling import (
    HARDWARE_NAMES,
    PHASE_LABELS,
    PHASE_ORDER,
    _dedup_latest,
    _is_instrumented_hgb,
    _phase_breakdown_ms,
    _thread_count,
    _workload_name,
    _workload_size,
)
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
    SOFTWARE_TEMPLATE,
    format_duration_ms,
    phase_variant_speedup_plot_html,
    render_hardware_tabs,
    render_software_tabs,
    variant_color_map,
)
from sklbench.reporting.matching import BenchmarkRecord, date_range, read_benchmark_records


SKLEARN_DEV_PIXI_ENV = "sklearn-dev"

# `_phase_breakdown_ms()` already carries the wall-clock fit time as
# "total_ms" alongside the real phases - reusing that dict key as one more
# entry in the phase loop below gets an "overall" column for free, appended
# after the real phases so it reads as their sum/grand-total.
TOTAL_PHASE_KEY = "total_ms"
PLOT_PHASE_ORDER = [*PHASE_ORDER, TOTAL_PHASE_KEY]
PLOT_PHASE_LABELS = {**PHASE_LABELS, TOTAL_PHASE_KEY: "overall"}


def _is_sklearn_dev_build(build_name: str) -> bool:
    return build_name == SKLEARN_DEV_PIXI_ENV or build_name.startswith(
        f"{SKLEARN_DEV_PIXI_ENV}@"
    )


def _base_build(builds) -> str | None:
    """The `sklearn-dev@<owner>:main` build among `builds`, if any. Picked
    dynamically rather than a fixed constant since `<owner>` varies with
    whichever fork/remote a branch comparison was run against (see
    CONTRIBUTING.md's `env@owner:ref` workflow)."""
    return next((build for build in builds if build.rsplit(":", 1)[-1] == "main"), None)


def _workload_data(
    records: list[BenchmarkRecord],
) -> tuple[
    dict[str, dict[int, dict[str, float]]],
    dict[str, tuple[int, int]],
    dict[str, dict],
]:
    """`{workload: {threads: {phase: ms}}}`, `{workload: (n_samples,
    n_features)}` and `{workload: estimator_params}` - the latter for
    hover-tooltip context, since e.g. "L" vs "L-leaf255" only differ by
    `max_leaf_nodes`/`max_iter`, not by name alone."""
    by_threads: dict[str, dict[int, dict[str, float]]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    hparams: dict[str, dict] = {}
    for record in records:
        breakdown = _phase_breakdown_ms(record)
        threads = _thread_count(record)
        if breakdown is None or threads is None:
            continue
        name = _workload_name(record)
        by_threads.setdefault(name, {})[threads] = breakdown
        sizes.setdefault(name, _workload_size(record))
        hparams.setdefault(name, record.case.get("algorithm", {}).get("estimator_params", {}))
    return by_threads, sizes, hparams


_HPARAMS_HOVER_EXCLUDED = {"early_stopping", "max_bins"}


def _format_hparams(hparams: dict) -> str:
    return ", ".join(
        f"{key}={value}"
        for key, value in sorted(hparams.items())
        if key not in _HPARAMS_HOVER_EXCLUDED
    )


def _hover_html(
    *,
    variant_build: str,
    workload: str,
    shape: tuple[int, int],
    threads: int,
    hparams: dict,
    normalized: float,
    absolute_ms: float,
    relative: float,
) -> str:
    n_samples, n_features = shape
    sign = "+" if absolute_ms >= 0 else "-"
    absolute_repr = f"{sign}{format_duration_ms(abs(absolute_ms))}"
    return "<br>".join(
        [
            f"<b>{escape(workload)} ({n_samples:,} x {n_features}), {threads} threads</b>",
            escape(_format_hparams(hparams)),
            f"normalized speed-up: {normalized:+.1%} of {escape(variant_build)}'s total fit time",
            f"absolute speed-up: {absolute_repr}",
            f"relative speed-up: {relative:.2f}x",
        ]
    )


def _collect_points(
    baseline_data: tuple[dict, dict, dict],
    variant_build: str,
    variant_data: tuple[dict, dict, dict],
) -> list[dict]:
    baseline_by_threads = baseline_data[0]
    variant_by_threads, variant_sizes, variant_hparams = variant_data

    points = []
    for name, threads_to_phases in variant_by_threads.items():
        base_threads_to_phases = baseline_by_threads.get(name)
        if not base_threads_to_phases:
            continue
        shape = variant_sizes[name]
        hparams = variant_hparams[name]
        for threads, variant_phases in threads_to_phases.items():
            baseline_phases = base_threads_to_phases.get(threads)
            if not baseline_phases:
                continue
            variant_total_ms = variant_phases["total_ms"]
            if variant_total_ms <= 0:
                continue
            for phase in PLOT_PHASE_ORDER:
                variant_ms = variant_phases[phase]
                if variant_ms <= 0:
                    continue
                baseline_ms = baseline_phases[phase]
                normalized = (baseline_ms - variant_ms) / variant_total_ms
                relative = baseline_ms / variant_ms
                # The "overall" pseudo-phase reads off its own axis (see
                # phase_variant_speedup_plot_html's secondary_axis_phase) as
                # a plain ratio, since summed-phase normalization isn't the
                # most legible way to read a single grand-total number.
                y = relative if phase == TOTAL_PHASE_KEY else normalized
                points.append(
                    {
                        "phase": phase,
                        "variant": variant_build,
                        "threads": threads,
                        "y": y,
                        "hover": _hover_html(
                            variant_build=variant_build,
                            workload=name,
                            shape=shape,
                            threads=threads,
                            hparams=hparams,
                            normalized=normalized,
                            absolute_ms=baseline_ms - variant_ms,
                            relative=relative,
                        ),
                    }
                )
    return points


def _software_summary(build_name: str, record: BenchmarkRecord) -> dict:
    summary = summarize_software_env(
        read_env("software", record.software_hash),
        record.implementation,
        software_hash=record.software_hash,
    )
    # Baseline and every variant are typically the same `implementation`
    # (e.g. plain "sklearn"), so the default name would collide across
    # builds - override it with the actual build label being compared.
    summary["name"] = build_name
    return summary


def render_hardware_page(records: list[BenchmarkRecord]) -> str:
    if not records:
        return '<section class="empty">No instrumented HGB results for this hardware.</section>'

    by_build: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        build = software_build_name(record.software_hash)
        if _is_sklearn_dev_build(build):
            by_build[build].append(record)

    base_build = _base_build(by_build)
    if base_build is None:
        available = ", ".join(sorted(by_build)) or "none"
        return (
            f'<section class="empty">No {SKLEARN_DEV_PIXI_ENV}@...:main baseline '
            f"results for this hardware (available sklearn-dev builds: "
            f"{available}).</section>"
        )
    variant_builds = sorted(build for build in by_build if build != base_build)
    if not variant_builds:
        return (
            f'<section class="empty">Only the baseline build ({base_build}) '
            "has results for this hardware - nothing to compare it against.</section>"
        )

    baseline_data = _workload_data(by_build[base_build])
    points = [
        point
        for variant_build in variant_builds
        for point in _collect_points(
            baseline_data, variant_build, _workload_data(by_build[variant_build])
        )
    ]
    if not points:
        return (
            '<section class="empty">No overlapping (workload, thread count) '
            f"points between {BASE_BUILD!r} and any variant build.</section>"
        )

    variant_colors = variant_color_map(variant_builds)
    by_threads: dict[int, list[dict]] = defaultdict(list)
    for point in points:
        by_threads[point["threads"]].append(point)

    cells = []
    for threads in sorted(by_threads):
        plot = phase_variant_speedup_plot_html(
            by_threads[threads],
            phase_order=PLOT_PHASE_ORDER,
            phase_labels=PLOT_PHASE_LABELS,
            variant_colors=variant_colors,
            secondary_axis_phase=TOTAL_PHASE_KEY,
        )
        cells.append(f'<section class="plot-cell"><h3>{threads} threads</h3>{plot}</section>')
    grid = (
        '<section class="plot-grid" '
        'style="grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));">'
        + "".join(cells)
        + "</section>"
    )

    software_tabs = render_software_tabs(
        [
            SOFTWARE_TEMPLATE.render(**_software_summary(base_build, by_build[base_build][0])),
            *(
                SOFTWARE_TEMPLATE.render(**_software_summary(build, by_build[build][0]))
                for build in variant_builds
            ),
        ],
        variant_colors=variant_colors,
    )

    rows = [
        DATE_RANGE_TEMPLATE.render(**date_range(records)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", records[0].hardware_hash))),
        software_tabs,
        grid,
    ]
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


if __name__ == "__main__":
    records = _dedup_latest(
        [record for record in read_benchmark_records() if _is_instrumented_hgb(record)]
    )
    by_hardware: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        by_hardware[record.hardware_hash].append(record)

    pages = [
        (HARDWARE_NAMES.get(hardware_hash, hardware_hash), render_hardware_page(hw_records))
        for hardware_hash, hw_records in sorted(by_hardware.items())
    ]

    html = BASE_TEMPLATE.render(
        title="HGB speed-up breakdown (branch comparison)",
        rows=[render_hardware_tabs(pages)],
    )
    output = dashboard_output_path("hgb_speedup_breakdown.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
