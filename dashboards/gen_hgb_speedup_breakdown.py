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
`sklearn-mkl`/`sklearn-openblas-openmp` against plain `sklearn`, and
excludes `sklearn-dev` builds for the same reason this dashboard excludes
everything else), so builds outside `sklearn-dev` (different BLAS/OpenMP
runtime, not different sklearn source) are excluded rather than folded in as
more variants. The baseline is picked dynamically as whichever `sklearn-dev`
build's ref is `main` (the `<owner>` half varies depending on which
fork/remote the comparison run
used), rather than a fixed constant.
"""
from collections import defaultdict
from html import escape
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.gen_hgb_scaling import (
    HARDWARE_NAMES,
    PHASE_LABELS,
    PHASE_ORDER,
    _dedup_latest,
    _has_active_wait,
    _is_instrumented_hgb,
    _phase_breakdown_ms,
    _raw_bench_env,
    _thread_count,
    _workload_name,
    _workload_size,
)
from dashboards.output import dashboard_output_path
from sklbench.reporting.envs import (
    active_wait_label_suffix,
    is_sklearn_dev_build,
    openmp_runtime_family,
    read_env,
    SKLEARN_DEV_PIXI_ENV,
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


_OPENMP_FAMILY_SHORT = {
    "GNU libgomp": "libgomp",
    "Intel/LLVM OpenMP": "libomp",
}

# `_phase_breakdown_ms()` already carries the wall-clock fit time as
# "total_ms" alongside the real phases - reusing that dict key as one more
# entry in the phase loop below gets an "overall" column for free, appended
# after the real phases so it reads as their sum/grand-total.
TOTAL_PHASE_KEY = "total_ms"
PLOT_PHASE_ORDER = [*PHASE_ORDER, TOTAL_PHASE_KEY]
PLOT_PHASE_LABELS = {**PHASE_LABELS, TOTAL_PHASE_KEY: "overall"}


def _has_sklearn_dev_build(records: list[BenchmarkRecord]) -> bool:
    return any(
        is_sklearn_dev_build(software_build_name(record.software_hash)) for record in records
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


def _relative_speedup_uncertainty(
    baseline_repeats: list[float], variant_repeats: list[float]
) -> float:
    """Rough absolute uncertainty on `median(baseline)/median(variant)`,
    propagated from each side's repeat-to-repeat spread as independent
    relative errors. Median-absolute-deviation (not stdev) to stay
    consistent with `_phase_breakdown_ms`'s own median aggregation - both
    are robust to the occasional stalled repeat this hardware sees, rather
    than treating it as real signal."""

    def relative_mad(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        center = median(values)
        if center <= 0:
            return 0.0
        return median(abs(value - center) for value in values) / center

    return math.sqrt(
        relative_mad(baseline_repeats) ** 2 + relative_mad(variant_repeats) ** 2
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
    is_overall: bool = False,
    relative_error: float = 0.0,
) -> str:
    n_samples, n_features = shape
    sign = "+" if absolute_ms >= 0 else "-"
    absolute_repr = f"{sign}{format_duration_ms(abs(absolute_ms))}"
    # The "overall" pseudo-phase reads its ratio off its own axis (see
    # phase_variant_speedup_plot_html's secondary_axis_phase) rather than as
    # a share of the variant's total fit time, so the normalized-speed-up
    # line (which is exactly that share) doesn't apply to it; repeat noise
    # is shown instead, since it's real dataset points/measurements rather
    # than a summed quantity.
    lines = [
        f"<b>{escape(workload)} ({n_samples:,} x {n_features}), {threads} threads</b>",
        escape(_format_hparams(hparams)),
    ]
    if not is_overall:
        lines.append(
            f"normalized speed-up: {normalized:+.1%} of {escape(variant_build)}'s total fit time"
        )
    lines.append(f"absolute speed-up: {absolute_repr}")
    lines.append(
        f"relative speed-up: {relative:.2f}x (±{relative_error:.2f}x repeat noise)"
        if is_overall
        else f"relative speed-up: {relative:.2f}x"
    )
    return "<br>".join(lines)


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
                is_overall = phase == TOTAL_PHASE_KEY
                y = relative if is_overall else normalized
                relative_error = (
                    relative
                    * _relative_speedup_uncertainty(
                        baseline_phases.get("total_ms_repeats") or [baseline_ms],
                        variant_phases.get("total_ms_repeats") or [variant_ms],
                    )
                    if is_overall
                    else 0.0
                )
                points.append(
                    {
                        "phase": phase,
                        "variant": variant_build,
                        "threads": threads,
                        "y": y,
                        **({"error": relative_error} if is_overall else {}),
                        "hover": _hover_html(
                            variant_build=variant_build,
                            workload=name,
                            shape=shape,
                            threads=threads,
                            hparams=hparams,
                            normalized=normalized,
                            absolute_ms=baseline_ms - variant_ms,
                            relative=relative,
                            is_overall=is_overall,
                            relative_error=relative_error,
                        ),
                    }
                )
    return points


def _software_summary(build_name: str, record: BenchmarkRecord) -> dict:
    summary = summarize_software_env(
        read_env("software", record.software_hash),
        record.implementation,
        software_hash=record.software_hash,
        case_env=_raw_bench_env(record),
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
        if is_sklearn_dev_build(build):
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

    plotted_records = [record for build_records in by_build.values() for record in build_records]
    rows = [
        DATE_RANGE_TEMPLATE.render(**date_range(plotted_records)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", records[0].hardware_hash))),
        software_tabs,
        grid,
    ]
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


def _env_key(record: BenchmarkRecord) -> tuple[str, str, bool]:
    return (
        record.hardware_hash,
        openmp_runtime_family(record.software_hash),
        _has_active_wait(record),
    )


def _env_label(hardware_hash: str, openmp_family: str, active_wait: bool) -> str:
    hardware_label = HARDWARE_NAMES.get(hardware_hash, hardware_hash)
    family_label = _OPENMP_FAMILY_SHORT.get(openmp_family, openmp_family)
    return f"{hardware_label} ({family_label}){active_wait_label_suffix(active_wait)}"


if __name__ == "__main__":
    records = _dedup_latest(
        [record for record in read_benchmark_records() if _is_instrumented_hgb(record)]
    )
    # Tabs are one per (hardware, OpenMP runtime family, active-wait) combo -
    # baseline and variants are only ever compared within the same combo (see
    # `render_hardware_page`), since a build swap that also swaps OpenMP
    # runtime or busy-wait behavior wouldn't isolate the sklearn-side change
    # the branch comparison is meant to show.
    by_env: dict[tuple[str, str, bool], list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        by_env[_env_key(record)].append(record)

    # Skip tabs with no sklearn-dev build at all - `render_hardware_page`'s
    # "no baseline" message already covers a sklearn-dev build without a
    # `main` to compare against, but an empty tab for a combo that was never
    # a branch-comparison run in the first place (e.g. libomp/MKL-only
    # combos) is just noise.
    pages = [
        (_env_label(*key), render_hardware_page(env_records))
        for key, env_records in sorted(by_env.items(), key=lambda item: _env_label(*item[0]))
        if _has_sklearn_dev_build(env_records)
    ]

    html = BASE_TEMPLATE.render(
        title="HGB speed-up breakdown (branch comparison)",
        rows=[render_hardware_tabs(pages)],
    )
    output = dashboard_output_path("hgb_speedup_breakdown.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
