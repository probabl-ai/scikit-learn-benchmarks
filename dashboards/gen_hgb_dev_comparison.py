"""The PR comparison table (see index_comparison.py) over the committed HGB
scalability results of `sklearn-dev*` builds: each branch against `main`,
for every machine and thread setting at once.
"""
from pathlib import Path

from dashboards import HARDWARE_NAMES
from dashboards.gen_hgb_scalability_breakdown import SOURCE_CONFIGS
from dashboards.index_comparison import (
    _base_build,
    _branch_label,
    _case_key,
    _env_of,
    _is_sklearn_dev_build,
    result_matches,
)
from sklbench.reporting.envs import software_build_name
from sklbench.reporting.html import BASE_TEMPLATE, DATE_RANGE_TEMPLATE, detailed_results_table_html
from sklbench.reporting.matching import (
    date_range,
    find_matches,
    matches_source_configs,
    read_all_results,
    read_failed_records,
)
from sklbench.reporting.utils import groupby


TITLE = "[dev] HGB branch vs main table"

ABOUT_HTML = """<section class="panel">
  <p>This table compares the fit and predict times of scikit-learn branches
  with <code>main</code>, for the HistGradientBoosting thread-scalability
  configs. It is the PR comparison table, with every machine in one table.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>There is one row per branch, machine, build and case. The
    <code>OMP_NUM_THREADS</code> column is the thread count, and the other
    env columns are the OpenMP settings of the run (active wait,
    <code>OMP_PROC_BIND</code>). Speed-ups are relative to <code>main</code>
    on the same machine with the same build and settings. Only branch rows are
    shown at first: clear the branch name filter to see the <code>main</code>
    rows too.</p>
  </details>
</section>"""


def _hardware_label(result) -> str:
    return HARDWARE_NAMES.get(result.hardware_hash, result.hardware_hash)


def _build_name(result) -> str:
    return software_build_name(result.software_hash)


def generate(output_dir: Path) -> None:
    results = [
        result
        for result in read_all_results()
        if matches_source_configs(result.case, SOURCE_CONFIGS)
        and _is_sklearn_dev_build(_build_name(result))
    ]
    failed_records = [
        record
        for record in read_failed_records()
        if matches_source_configs(record.case, SOURCE_CONFIGS)
        and _is_sklearn_dev_build(_build_name(record))
    ]

    matches = []
    unmatched_base_results = []
    unmatched_variant_results = []
    failed_for_table = []
    # The match key ignores the machine, so match within each machine and
    # build env only.
    results_by_group = groupby(results, lambda r: (r.hardware_hash, _env_of(_build_name(r))))
    failed_by_group = groupby(failed_records, lambda r: (r.hardware_hash, _env_of(_build_name(r))))
    for group in sorted(set(results_by_group) | set(failed_by_group)):
        by_build = groupby(results_by_group.get(group, []), _build_name)
        failed_by_build = groupby(failed_by_group.get(group, []), _build_name)
        base_build = _base_build(sorted(set(by_build) | set(failed_by_build)))
        if base_build is None:
            continue
        base_results = by_build.get(base_build, [])
        base_by_case_key = groupby(base_results, lambda r: _case_key(r.case))
        base_failed_keys = {_case_key(record.case) for record in failed_by_build.get(base_build, [])}
        failed_for_table.extend(
            (record, _branch_label(base_build)) for record in failed_by_build.get(base_build, [])
        )
        for build in sorted(set(by_build) | set(failed_by_build)):
            if build == base_build:
                continue
            variant_results = by_build.get(build, [])
            matches.extend(find_matches(base_results, variant_results, result_matches))
            # Keep the side that succeeded when the other one failed.
            unmatched_variant_results.extend(
                r for r in variant_results if _case_key(r.case) in base_failed_keys
            )
            for record in failed_by_build.get(build, []):
                failed_for_table.append((record, _branch_label(build)))
                unmatched_base_results.extend(base_by_case_key.get(_case_key(record.case), []))

    variant_builds = {_build_name(match.matched_result) for match in matches}
    latest_variant = max(
        (match.matched_result for match in matches),
        key=lambda result: result.timestamp_recorded,
        default=None,
    )
    table = detailed_results_table_html(
        "all",
        groupby(matches, lambda match: match.matched_result.method),
        baseline_label=lambda result: _branch_label(_build_name(result)),
        variant_label=lambda result: _branch_label(_build_name(result)),
        failed_records=failed_for_table,
        unmatched_base_results=unmatched_base_results,
        unmatched_candidate_results=unmatched_variant_results,
        collapsible=False,
        variant_column_title="Branch name",
        hardware_label=_hardware_label,
        env_label=lambda result: _env_of(_build_name(result)),
        default_variant_filter=(
            _branch_label(_build_name(latest_variant))
            if latest_variant is not None and len({_branch_label(b) for b in variant_builds}) == 1
            else None
        ),
    )
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[
            ABOUT_HTML,
            DATE_RANGE_TEMPLATE.render(**date_range(list({id(r.record): r.record for r in results}.values()))),
            table or '<section class="empty">No branch has been compared with main yet.</section>',
        ],
    )
    output = output_dir / "hgb_dev_comparison.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
