from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.utils import (
    partition_iterable, groupby, stable_json, without_keys,
)

from sklbench.reporting.matching import (
    append_iterations_warning, read_all_results, read_failed_records,
    find_matches, date_range, BenchmarkRecord, Match, MatchWarning, MethodResult,
    is_scaling_benchmark,
)

from sklbench.reporting.envs import (
    is_vanilla_sklearn, read_env, software_build_name, summarize_software_env,
    summarize_hardware_env,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    assemble_plots_in_grid,
    detailed_results_table_html,
    speedup_plot_html,
    render_software_tabs,
    render_hardware_tabs,
    variant_color_map,
)


HARDWARE_NAMES = {
    "534824": "Intel GNR 172 CPU cores",
    "3b5e61": "Intel laptop with B390 GPU",
}
PIXI_ENV = "sklearn-dev"
BASE_IMPLEMENTATION = "sklearn"
# "sklearn-conda" was the pixi environment's name before its rename to
# "sklearn-cf-default"; results captured before that rename still record the
# old name (see sklbench.reporting.envs.VANILLA_SKLEARN_PIXI_ENVS for the
# analogous sklearn/sklearn-pypi case). Included here as an extra candidate,
# as a sanity check that it tracks the vanilla pypi baseline.
CONDA_FORGE_PIXI_ENVS = {"sklearn-cf-default", "sklearn-conda"}


def _pixi_environment_name(result: MethodResult | BenchmarkRecord) -> str:
    return read_env("software", result.software_hash)["pixi_environment_name"]


def is_included_result(result: MethodResult | BenchmarkRecord) -> bool:
    """A pure sklearn build relevant to this comparison: the vanilla pypi
    baseline, a `sklearn-dev` scikit-learn git checkout (labelled by ref, e.g.
    `1.9`/`main`), or a `sklearn-cf-default` conda-forge build (as opposed to
    sklearnex/array API implementations, or other sklearn builds not part of
    this comparison)."""
    if result.implementation.short_name != BASE_IMPLEMENTATION:
        return False
    if is_vanilla_sklearn(result.software_hash):
        return True
    build_name = software_build_name(result.software_hash)
    return (
        build_name.startswith(f"{PIXI_ENV}@")
        or _pixi_environment_name(result) in CONDA_FORGE_PIXI_ENVS
    )


def is_baseline_result(result: MethodResult | BenchmarkRecord) -> bool:
    return is_included_result(result) and is_vanilla_sklearn(result.software_hash)


def match_build_variant(match: Match) -> str:
    return software_build_name(match.matched_result.software_hash)


def _dedup_latest_by_variant(results: list[MethodResult]) -> list[MethodResult]:
    """Keep only the most recent run per (build variant, case). A build can be
    re-run days apart (e.g. sklearn-cf-default) - without this, every run
    would show up as a separate point for the same case, unlike the baseline
    side which find_matches already dedups this way."""
    latest: dict[tuple[str, str], MethodResult] = {}
    for result in results:
        key = (software_build_name(result.software_hash), result.minimal_match_key)
        current = latest.get(key)
        if current is None or result.timestamp_recorded > current.timestamp_recorded:
            latest[key] = result
    return list(latest.values())


def _case_key(case: dict) -> str:
    """Case identity ignoring implementation/max_bins - shared by a base result
    and the candidate(s) it would be compared against (or vice versa)."""
    return stable_json(without_keys(case, excluded_names={"implementation", "max_bins"}))


def result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list[MatchWarning]]:
    """
    Assumptions:
    - hardware matches
    - both base_res and candidate are pure sklearn builds included in this
      comparison, base_res is the vanilla pypi baseline build

    returns:
    - True/False
    - warnings
    """
    assert base_res.hardware_hash == candidate.hardware_hash
    assert is_included_result(base_res) and is_included_result(candidate)

    warnings = []
    append_iterations_warning(base_res, candidate, warnings)

    return (
        base_res.minimal_match_key == candidate.minimal_match_key,
        warnings
    )


def render_hardware_page(
    results: list[MethodResult],
    failed_records: list[BenchmarkRecord],
    hardware_hash: str,
) -> str:
    results = [
        res for res in results
        if res.hardware_hash == hardware_hash and is_included_result(res)
    ]
    failed_records = [
        record for record in failed_records
        if record.hardware_hash == hardware_hash and is_included_result(record)
    ]
    if not results:
        return '<section class="empty">No benchmark results for this hardware.</section>'
    hardwares_set = {res.hardware_hash for res in results}
    if len(hardwares_set) > 1:
        raise ValueError(f"Results are dirty: several hardware hashes match {hardware_hash!r}")

    base_results, other_results = partition_iterable(
        results, predicate=is_baseline_result
    )
    if not base_results:
        return f'<section class="empty">No vanilla {BASE_IMPLEMENTATION} baseline results for this hardware.</section>'
    other_results = _dedup_latest_by_variant(other_results)
    baseline_label = software_build_name(base_results[0].software_hash)

    variant_colors = variant_color_map(
        sorted({software_build_name(res.software_hash) for res in other_results})
    )
    grouped_results = groupby(base_results, lambda res: (res.category, res.method))

    # Non-baseline (candidate) failures: shown on the plots too, at the bottom
    # of their model-variant column, since we don't know from a failed record
    # whether fit or predict is what failed.
    candidate_failed_records = [
        record for record in failed_records if not is_baseline_result(record)
    ]
    candidate_failed_by_category = groupby(
        candidate_failed_records, lambda record: record.category
    )

    plots = []
    matches_by_category = {}
    for (category, method), group_base_results in grouped_results.items():
        matches = find_matches(group_base_results, other_results, result_matches)
        matches_by_category.setdefault(category, {})[method] = matches
        # create a JS snippet for plotly:
        plots.append({
            "category": category,
            "method": method,
            "point_count": len(matches),
            "plot": speedup_plot_html(
                matches,
                baseline_label=baseline_label,
                variant_colors=variant_colors,
                trace_variant=match_build_variant,
                x_variant=match_build_variant,
                failed_records=candidate_failed_by_category.get(category, []),
            )
        })
    failed_by_category = groupby(failed_records, lambda record: record.category)

    # A failed record means find_matches never sees a pair for that case, so the
    # side that *did* succeed - the base when a candidate failed, or any
    # candidate when the base itself failed - would otherwise silently vanish
    # from the table too. Look those up by case identity so they still show up
    # (with no speedup, since there's nothing successful to compare against).
    base_by_case_key: dict[str, list[MethodResult]] = {}
    for base in base_results:
        base_by_case_key.setdefault(_case_key(base.case), []).append(base)
    other_by_case_key: dict[str, list[MethodResult]] = {}
    for other in other_results:
        other_by_case_key.setdefault(_case_key(other.case), []).append(other)

    unmatched_base_by_category: dict[str, list[MethodResult]] = {}
    unmatched_candidate_by_category: dict[str, list[MethodResult]] = {}
    for record in failed_records:
        key = _case_key(record.case)
        if is_baseline_result(record):
            for candidate in other_by_case_key.get(key, []):
                unmatched_candidate_by_category.setdefault(candidate.category, []).append(candidate)
        else:
            for base in base_by_case_key.get(key, []):
                unmatched_base_by_category.setdefault(base.category, []).append(base)

    details_by_category = {
        category: detailed_results_table_html(
            category,
            matches_by_category.get(category, {}),
            baseline_label=baseline_label,
            variant_label=lambda result: software_build_name(result.software_hash),
            failed_records=[
                (record, software_build_name(record.software_hash))
                for record in failed_by_category.get(category, [])
            ],
            unmatched_base_results=unmatched_base_by_category.get(category, []),
            unmatched_candidate_results=unmatched_candidate_by_category.get(category, []),
        )
        for category in (
            set(matches_by_category) | set(failed_by_category)
            | set(unmatched_base_by_category) | set(unmatched_candidate_by_category)
        )
    }

    hardware_hash, = hardwares_set
    hardware_env = read_env("hardware", hardware_hash)

    base_sw_env = read_env("software", base_results[0].software_hash)
    base_implem = base_results[0].implementation

    base_summary = summarize_software_env(
        base_sw_env,
        base_implem,
        software_hash=base_results[0].software_hash,
    )
    base_summary["name"] = baseline_label
    softwares = [base_summary]
    for build_name, implem_results in groupby(
        other_results, lambda res: software_build_name(res.software_hash)
    ).items():
        res = implem_results[0]
        env = read_env("software", res.software_hash)
        summary = summarize_software_env(
            env,
            res.implementation,
            software_hash=res.software_hash,
        )
        summary["name"] = build_name
        softwares.append(summary)

    rows = [
        DATE_RANGE_TEMPLATE.render(date_range(results)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(hardware_env)),
        render_software_tabs([
            SOFTWARE_TEMPLATE.render(**summary)
            for summary in softwares
        ], variant_colors=variant_colors),
        assemble_plots_in_grid(
            plots,
            rows={"category": ["linear", "tree-based", "clustering"]},
            columns={"method": ["fit", "predict"]},
            details_by_row=details_by_category,
        )
    ]
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


if __name__ == "__main__":
    results = [res for res in read_all_results() if not is_scaling_benchmark(res)]
    failed_records = [
        record for record in read_failed_records() if not is_scaling_benchmark(record)
    ]
    hardware_pages = [
        (hardware_name, render_hardware_page(results, failed_records, hardware_hash))
        for hardware_hash, hardware_name in HARDWARE_NAMES.items()
    ]

    html = BASE_TEMPLATE.render(
        title="sklbench sklearn-dev builds comparison dashboard",
        rows=[
            render_hardware_tabs(hardware_pages),
        ],
    )

    output = dashboard_output_path("sklearn_dev_comparison.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
