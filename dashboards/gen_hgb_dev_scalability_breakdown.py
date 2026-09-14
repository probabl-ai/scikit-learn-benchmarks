"""HistGradientBoosting fit-time phase breakdown across thread counts,
restricted to `sklearn-dev*` builds - branch vs `main` side by side wherever
both are present for the same hardware/runtime combo.

Reuses the instrumented-HGB record selection, phase-breakdown math and
per-workload rendering from gen_hgb_scalability_breakdown.py (see that module's docstring
for why raw `read_benchmark_records()` is read and how thread count/dedup
identity work) - this dashboard differs only in which builds it includes and
in comparing builds against each other rather than showing one build per tab.

`sklearn-dev*` builds are one-off scikit-learn git-checkout builds (a
specific commit/PR branch, see CONTRIBUTING.md's `setup_sklearn_ref.sh` /
`run.sh env@owner:ref` workflow) rather than a stable environment build, so
a "software build" tab here means "whatever ref that checkout was on", not a
fixed BLAS/OpenMP variant. gen_hgb_scalability_breakdown.py excludes these builds for the
same reason, in the other direction - kept split into two dashboards rather
than one so neither's per-build tabs mix apples (stable builds) with oranges
(one-off dev checkouts).

Tabs group by (hardware, OpenMP runtime family, active-wait, proc-bind)
rather than by `software_hash` (unlike gen_hgb_scalability_breakdown.py's tabs) -
see gen_hgb_dev_speedup_breakdown.py's `_env_key` for the same reasoning: a
branch build and the `main` build it's meant to be compared against are two
different `software_hash`es, so grouping by the exact hash would split them
across tabs instead of landing them on the same page/plot.
"""
from dashboards.gen_hgb_scalability_breakdown import (
    HARDWARE_NAMES,
    PHASE_COLORS,
    PHASE_LABELS,
    PHASE_ORDER,
    _dedup_latest,
    _hardware_sort_index,
    _has_active_wait,
    _is_instrumented_hgb,
    _is_sklearn_dev_build,
    _legend_html,
    _phase_breakdown_ms,
    _proc_bind,
    _raw_bench_env,
    _subtitle_html,
    _thread_count,
    _tree_n_threads,
    _workload_name,
    _workload_size,
    render_env_page,
)
from dashboards import dashboard_output_path
from sklbench.reporting.envs import (
    active_wait_label_suffix,
    OPENMP_FAMILY_SHORT_LABELS,
    openmp_runtime_family,
    proc_bind_label_suffix,
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
    phase_breakdown_plot_html,
    render_hardware_tabs,
    render_software_tabs,
)
from sklbench.reporting.matching import BenchmarkRecord, date_range, read_benchmark_records


def _base_build(builds) -> str | None:
    """The `sklearn-dev@<owner>:main` build among `builds`, if any - see
    gen_hgb_dev_speedup_breakdown.py's `_base_build` for why this is picked
    dynamically (by ref name) rather than a fixed constant."""
    return next((build for build in builds if build.rsplit(":", 1)[-1] == "main"), None)


def _single_variant_build(by_build: dict[str, list[BenchmarkRecord]], base_build: str) -> str | None:
    """The one non-`main` build to compare against `base_build` - when more
    than one is present (e.g. several branches benchmarked on the same
    hardware over time), keep only the most recently recorded. This
    dashboard is a plain two-way branch-vs-main comparison, not a many-way
    one (see gen_hgb_dev_speedup_breakdown.py for that instead), so extra
    older branches would just clutter the bars rather than add signal."""
    variants = [build for build in by_build if build != base_build]
    if not variants:
        return None
    return max(
        variants, key=lambda build: max(record.timestamp_recorded for record in by_build[build])
    )


def _series_label(build: str, base_build: str) -> str:
    """Short per-series label for a build's bars/x-axis subcategory: plain
    "main" for the baseline build, else the `owner:ref` half of
    `software_build_name`'s `sklearn-dev@owner:ref` - the shared
    `sklearn-dev@`/pixi-env prefix is implied (every series on this
    dashboard is a sklearn-dev build already) and would just be noise
    repeated on every bar."""
    if build == base_build:
        return "main"
    return build.rsplit("@", 1)[-1]


def _env_key(record: BenchmarkRecord) -> tuple[str, str, bool, str | None]:
    return (
        record.hardware_hash,
        openmp_runtime_family(record.software_hash),
        _has_active_wait(record),
        _proc_bind(record),
    )


def _env_label(hardware_hash: str, openmp_family: str, active_wait: bool, proc_bind: str | None) -> str:
    hardware_label = HARDWARE_NAMES.get(hardware_hash, hardware_hash)
    family_label = OPENMP_FAMILY_SHORT_LABELS.get(openmp_family, openmp_family)
    return (
        f"{hardware_label} ({family_label})"
        f"{active_wait_label_suffix(active_wait)}"
        f"{proc_bind_label_suffix(proc_bind)}"
    )


def _software_summary(build: str, record: BenchmarkRecord) -> dict:
    summary = summarize_software_env(
        read_env("software", record.software_hash),
        record.implementation,
        software_hash=record.software_hash,
        case_env=_raw_bench_env(record),
    )
    # Every build compared here is already a sklearn-dev checkout of the same
    # pixi env, differing only by git ref - name the tab/panel by that ref
    # (`build`, e.g. "sklearn-dev@cakedev0:my-branch") rather than the
    # generic `implementation.short_name` every build would otherwise share.
    summary["name"] = build
    return summary


def render_env_page_comparison(
    records: list[BenchmarkRecord], base_build: str, by_build: dict[str, list[BenchmarkRecord]]
) -> str:
    """Like `render_env_page`, but with `by_build`'s two builds (the variant
    and `base_build`) plotted as side-by-side bars per thread count instead
    of one page per build - the variant renders left of `main`, which always
    renders rightmost. X-axis ticks stay plain thread counts (see
    `phase_breakdown_plot_html`'s `series` handling) - only left/right bar
    position encodes which build is which."""
    build_order = [build for build in sorted(by_build) if build != base_build] + [base_build]
    series_order = [_series_label(build, base_build) for build in build_order]
    variant_build = build_order[0]

    by_workload: dict[str, dict[str, list[BenchmarkRecord]]] = {}
    sample_record: dict[str, BenchmarkRecord] = {}
    for build in build_order:
        for record in by_build[build]:
            name = _workload_name(record)
            by_workload.setdefault(name, {}).setdefault(build, []).append(record)
            sample_record.setdefault(name, record)

    def _x_label(threads: int, tree_n_threads: int | None) -> str:
        return f"{threads} ({tree_n_threads})" if tree_n_threads is not None else str(threads)

    cells = []
    for name in sorted(by_workload, key=lambda n: _workload_size(sample_record[n])):
        # The actual tree-growing thread count can differ between the
        # variant and `main` at the same requested thread count (e.g. a
        # branch that sizes threads down differently for small workloads) -
        # both builds' bars share one x tick per requested thread count
        # though, so the parenthetical always reflects the variant's actual
        # count (what's being evaluated), not whichever build happened to
        # run last in the loop below.
        variant_labels = {
            threads: _x_label(threads, _tree_n_threads(record))
            for record in by_workload[name].get(variant_build, [])
            if (threads := _thread_count(record)) is not None
        }
        points = []
        for build in build_order:
            series = _series_label(build, base_build)
            for record in by_workload[name].get(build, []):
                breakdown = _phase_breakdown_ms(record)
                threads = _thread_count(record)
                if breakdown is None or threads is None:
                    continue
                x_label = variant_labels.get(threads) or _x_label(threads, _tree_n_threads(record))
                points.append(
                    {
                        "x": threads,
                        "x_label": x_label,
                        "series": series,
                        "phases": breakdown,
                        "total_ms": breakdown["total_ms"],
                    }
                )
        if not points:
            continue
        n_samples, n_features = _workload_size(sample_record[name])
        title = f"{name} ({n_samples:,} x {n_features})"
        subtitle_html = _subtitle_html(sample_record[name])
        plot = phase_breakdown_plot_html(
            points,
            phase_order=PHASE_ORDER,
            phase_colors=PHASE_COLORS,
            phase_labels=PHASE_LABELS,
            series_order=series_order,
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
    software_tabs = render_software_tabs(
        [SOFTWARE_TEMPLATE.render(**_software_summary(build, by_build[build][0])) for build in build_order]
    )
    rows = [
        DATE_RANGE_TEMPLATE.render(**date_range(records)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", records[0].hardware_hash))),
        software_tabs,
        _legend_html() + grid,
    ]
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


if __name__ == "__main__":
    records = _dedup_latest(
        [
            record
            for record in read_benchmark_records()
            if _is_instrumented_hgb(record) and _is_sklearn_dev_build(record)
        ]
    )
    by_env: dict[tuple[str, str, bool, str | None], list[BenchmarkRecord]] = {}
    for record in records:
        by_env.setdefault(_env_key(record), []).append(record)

    pages = []
    for key, env_records in sorted(
        by_env.items(), key=lambda item: (_hardware_sort_index(item[0][0]), _env_label(*item[0]))
    ):
        by_build: dict[str, list[BenchmarkRecord]] = {}
        for record in env_records:
            by_build.setdefault(software_build_name(record.software_hash), []).append(record)
        base_build = _base_build(by_build)
        variant_build = _single_variant_build(by_build, base_build) if base_build is not None else None
        # Only worth comparing side by side once there's a `main` baseline
        # and one other build to line up against it - otherwise this is
        # exactly gen_hgb_scalability_breakdown.py's single-build page.
        if base_build is not None and variant_build is not None:
            compared_by_build = {base_build: by_build[base_build], variant_build: by_build[variant_build]}
            compared_records = compared_by_build[base_build] + compared_by_build[variant_build]
            page = render_env_page_comparison(compared_records, base_build, compared_by_build)
        else:
            page = render_env_page(env_records)
        pages.append((_env_label(*key), page))

    html = BASE_TEMPLATE.render(
        title="HGB fit-time breakdown (thread scalability, sklearn-dev)",
        rows=[render_hardware_tabs(pages)],
    )
    output = dashboard_output_path("hgb_dev_scaling.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
