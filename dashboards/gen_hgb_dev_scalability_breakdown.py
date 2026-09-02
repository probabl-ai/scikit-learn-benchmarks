"""HistGradientBoosting fit-time phase breakdown across thread counts,
restricted to `sklearn-dev*` builds.

Reuses the instrumented-HGB record selection, phase-breakdown math and
per-workload rendering from gen_hgb_scaling.py (see that module's docstring
for why raw `read_benchmark_records()` is read and how thread count/dedup
identity work) - this dashboard differs only in which builds it includes.

`sklearn-dev*` builds are one-off scikit-learn git-checkout builds (a
specific commit/PR branch, see CONTRIBUTING.md's `setup_sklearn_ref.sh` /
`run.sh env@owner:ref` workflow) rather than a stable environment build, so
a "software build" tab here means "whatever ref that checkout was on", not a
fixed BLAS/OpenMP variant. gen_hgb_scaling.py excludes these builds for the
same reason, in the other direction - kept split into two dashboards rather
than one so neither's per-build tabs mix apples (stable builds) with oranges
(one-off dev checkouts).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.gen_hgb_scalability_breakdown import (
    _dedup_latest,
    _env_key,
    _env_label,
    _is_instrumented_hgb,
    _is_sklearn_dev_build,
    render_env_page,
)
from dashboards.output import dashboard_output_path
from sklbench.reporting.html import BASE_TEMPLATE, render_hardware_tabs
from sklbench.reporting.matching import BenchmarkRecord, read_benchmark_records


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

    # Same tab structure as gen_hgb_scaling.py - one per (hardware, software
    # build, active-wait, proc-bind) combo, see that module's __main__ for why.
    pages = [
        (
            _env_label(hardware_hash, software_hash, active_wait, proc_bind),
            render_env_page(env_records),
        )
        for (hardware_hash, software_hash, active_wait, proc_bind), env_records in sorted(
            by_env.items(),
            key=lambda item: _env_label(*item[0]),
        )
    ]

    html = BASE_TEMPLATE.render(
        title="HGB fit-time breakdown (thread scalability, sklearn-dev)",
        rows=[render_hardware_tabs(pages)],
    )
    output = dashboard_output_path("hgb_dev_scaling.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
