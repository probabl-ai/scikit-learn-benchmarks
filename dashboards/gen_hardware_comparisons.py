"""GNR-vs-laptop hardware speed-up, for a curated set of build/implementation
variants - each variant compared against *itself* across the two machines,
never against a different variant.

Every match pairs the same `variant_label` (a plain-sklearn build name like
"sklearn-cf-mkl", or an implementation short name like "sklearnex-cpu") on
both hardwares, so hardware and build/implementation are never varied within
the same pairing - a single ratio for "GNR + sklearnex-cpu vs laptop +
sklearn-pypi" would conflate two different causes into one number. The three
variants (plus their laptop/GNR counterparts) are shown together, as
separate colored series, on one page - there's only one axis being compared
(hardware), so there's no need for per-variant tabs.

`CANDIDATE_LABELS` is a deliberately short, curated list rather than every
build this repo produces (see `pixi.toml`'s `[environments]` for the rest,
e.g. the libgomp/libomp/libomp-omp OpenBLAS variants) - those are covered by
`gen_builds_comparison.py` (laptop and GNR each get their own tab there)
instead; this dashboard's job is specifically the GNR/laptop hardware gap
for the variants folks actually reach for.
"""
from dataclasses import replace
from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    assemble_plots_in_grid,
    detailed_results_table_html,
    render_software_tabs,
    speedup_plot_html,
    variant_color_map,
)
from sklbench.reporting.envs import (
    read_env,
    software_build_name,
    summarize_hardware_env,
    summarize_software_env,
)
from sklbench.reporting.matching import (
    append_iterations_warning,
    append_max_bins_warning,
    find_matches,
    read_all_results,
    date_range,
    is_models_scalability_result,
    is_scaling_benchmark,
    BenchmarkRecord,
    Match,
    MatchWarning,
    MethodResult,
)
from sklbench.reporting.utils import stable_json, without_keys


HARDWARE_NAMES = {
    "3b5e61": "Intel laptop",
    "534824": "Intel GNR 172 CPU cores",
}
BASELINE_HARDWARE_HASH = "3b5e61"
CANDIDATE_HARDWARE_HASH = "534824"
BASELINE_LABEL = HARDWARE_NAMES[BASELINE_HARDWARE_HASH]
BASE_IMPLEMENTATION = "sklearn"
# Fixed display/sort order.
CANDIDATE_LABELS = ["sklearn-pypi", "sklearn-cf-mkl", "sklearnex-cpu"]
CATEGORIES = ["linear", "tree-based", "clustering"]
METHODS = ["fit", "predict"]


def variant_label(result: MethodResult | BenchmarkRecord) -> str:
    """The sklearn build name (e.g. "sklearn-cf-mkl") for plain-sklearn
    results, or the implementation short name (e.g. "sklearnex-cpu")
    otherwise - the axis this dashboard holds fixed while comparing
    hardware."""
    if result.implementation.short_name == BASE_IMPLEMENTATION:
        return software_build_name(result.software_hash)
    return result.implementation.short_name


def is_baseline_result(result: MethodResult | BenchmarkRecord) -> bool:
    return result.hardware_hash == BASELINE_HARDWARE_HASH and variant_label(result) in CANDIDATE_LABELS


def is_candidate_result(result: MethodResult | BenchmarkRecord) -> bool:
    return result.hardware_hash == CANDIDATE_HARDWARE_HASH and variant_label(result) in CANDIDATE_LABELS


# `n_jobs` and RF/ET's `n_estimators` are both derived in `real_datasets.py`
# from `N_JOBS = floor(0.9 * cpu_count(...))` - the *local* machine's core
# count at config-generation time - so they legitimately differ between the
# laptop and GNR runs of what's otherwise the identical case. Excluded from
# the match key for the same reason `n_jobs` already is: this comparison
# should still pair these up rather than treat them as different workloads.
_MATCH_EXCLUDED_NAMES = {"implementation", "max_bins", "n_jobs", "n_estimators"}

# Same normalization/target as gen_models_scalability.py's
# `NORMALIZED_N_ESTIMATORS`: RF/ET fit and predict time both scale
# ~linearly with forest size, and since that size differs between machines
# here (see `_MATCH_EXCLUDED_NAMES` above), comparing raw times would
# conflate "faster hardware" with "this pairing happened to fit/predict a
# bigger forest" - rescaling every RF/ET result to a common forest size
# divides that confound back out before speed-ups are computed.
NORMALIZED_N_ESTIMATORS = 100


def _n_estimators(result: MethodResult) -> int | None:
    return result.case.get("algorithm", {}).get("estimator_params", {}).get(
        "n_estimators"
    )


def _normalize_tree_result(result: MethodResult) -> MethodResult:
    n_estimators = _n_estimators(result)
    if not n_estimators:
        return result
    scale = NORMALIZED_N_ESTIMATORS / n_estimators
    return replace(result, times=[t * scale for t in result.times])


# Cross-hardware comparisons in this dashboard routinely pair up results
# whose model isn't actually identical - RF/ET's `n_estimators` varies by
# machine (see `_normalize_tree_result`) - so `Match.metrics_differences`
# and the iteration-count check in `append_iterations_warning` would just
# flag that expected divergence as a reliability warning on every other
# point. Clearing `metrics` and dropping the specific attributes those
# checks read (`has_onedal_estimator`, which also drives
# `is_sklearnex_fallback`'s "fell back to scikit-learn" marker; `n_iter`)
# suppresses that noise while leaving other attributes (e.g. `solver`,
# shown in the detailed table) untouched.
_DROPPED_ATTRIBUTES = {"has_onedal_estimator", "n_iter"}


def _drop_metrics_and_reliability_signals(result: MethodResult) -> MethodResult:
    attributes = {
        name: value
        for name, value in result.attributes.items()
        if name not in _DROPPED_ATTRIBUTES
    }
    return replace(result, metrics={}, attributes=attributes)


def _match_key(result: MethodResult) -> str:
    case = without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES)
    case["method"] = result.method
    return stable_json(case)


def _table_comparison_key(result: MethodResult) -> str:
    return stable_json(without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES))


def result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list[MatchWarning]]:
    assert base_res.hardware_hash == BASELINE_HARDWARE_HASH
    assert candidate.hardware_hash == CANDIDATE_HARDWARE_HASH
    assert variant_label(base_res) == variant_label(candidate)

    warnings = []
    # `append_max_bins_warning` assumes a vanilla-sklearn base compared
    # against a sklearnex candidate - true for the "sklearn-pypi"/
    # "sklearn-cf-mkl" labels, but for "sklearnex-cpu" both sides are
    # sklearnex (same variant, different hardware), so the warning doesn't
    # apply there.
    if base_res.implementation.library == BASE_IMPLEMENTATION and candidate.is_sklearnex_tree:
        append_max_bins_warning(base_res, candidate, warnings)
    append_iterations_warning(base_res, candidate, warnings)

    return _match_key(base_res) == _match_key(candidate), warnings


def match_variant_label(match: Match) -> str:
    return variant_label(match.matched_result)


def _comparison_page(rows: list[str]) -> str:
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


def _hardware_env_badges() -> list[str]:
    """One tab per hardware, independent of the software-envs tabs below -
    `HARDWARE_TEMPLATE` has no leading `<h3>{name}</h3>` of its own (unlike
    `SOFTWARE_TEMPLATE`), so one is prepended here for `render_software_tabs`
    to pick up as the tab label."""
    return [
        f"<h3>{escape(name)}</h3>"
        + HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hardware_hash)))
        for hardware_hash, name in HARDWARE_NAMES.items()
    ]


def _software_env_badges(
    baseline_results: list[MethodResult], candidate_results: list[MethodResult]
) -> list[str]:
    """One tab per build/implementation label - a single representative env
    (laptop's if present, else GNR's), since the two machines run the same
    pinned build."""
    badges = []
    for label in CANDIDATE_LABELS:
        source = next((r for r in baseline_results if variant_label(r) == label), None) or next(
            (r for r in candidate_results if variant_label(r) == label), None
        )
        if source is None:
            continue
        summary = summarize_software_env(
            read_env("software", source.software_hash),
            source.implementation,
            software_hash=source.software_hash,
        )
        summary["name"] = label
        badges.append(SOFTWARE_TEMPLATE.render(**summary))
    return badges


def render_comparison(all_results: list[MethodResult]) -> str:
    baseline_results = [
        _normalize_tree_result(result) for result in all_results if is_baseline_result(result)
    ]
    candidate_results = [
        _normalize_tree_result(result) for result in all_results if is_candidate_result(result)
    ]
    if not baseline_results or not candidate_results:
        return '<section class="empty">No overlapping laptop/GNR results for these builds.</section>'

    trace_colors = variant_color_map(CANDIDATE_LABELS)

    plots = []
    matches_by_category = {}
    for category in CATEGORIES:
        for method in METHODS:
            # One `find_matches` call per label so a laptop result for one
            # variant can never be paired against a GNR result for another -
            # `_match_key` deliberately excludes `implementation`, so that
            # cross-variant mismatch wouldn't otherwise be structurally
            # impossible.
            category_method_matches = [
                match
                for label in CANDIDATE_LABELS
                for match in find_matches(
                    [
                        result
                        for result in baseline_results
                        if result.category == category
                        and result.method == method
                        and variant_label(result) == label
                    ],
                    [
                        result
                        for result in candidate_results
                        if result.category == category
                        and result.method == method
                        and variant_label(result) == label
                    ],
                    result_matches,
                    match_key=_match_key,
                )
            ]
            matches_by_category.setdefault(category, {})[method] = category_method_matches
            plots.append(
                {
                    "category": category,
                    "method": method,
                    "point_count": len(category_method_matches),
                    "plot": speedup_plot_html(
                        category_method_matches,
                        baseline_label=BASELINE_LABEL,
                        variant_colors=trace_colors,
                        trace_variant=match_variant_label,
                        x_variant=match_variant_label,
                        variant_sort_key=CANDIDATE_LABELS.index,
                        comparison_key=_table_comparison_key,
                    ),
                }
            )

    return _comparison_page(
        [
            DATE_RANGE_TEMPLATE.render(date_range(baseline_results + candidate_results)),
            render_software_tabs(_hardware_env_badges()),
            render_software_tabs(
                _software_env_badges(baseline_results, candidate_results),
                variant_colors=trace_colors,
            ),
            assemble_plots_in_grid(
                plots,
                rows={"category": CATEGORIES},
                columns={"method": METHODS},
                details_by_row={
                    category: detailed_results_table_html(
                        category,
                        category_matches,
                        baseline_label=BASELINE_LABEL,
                        variant_label=variant_label,
                        comparison_key=_table_comparison_key,
                    )
                    for category, category_matches in matches_by_category.items()
                },
            ),
        ]
    )


if __name__ == "__main__":
    all_results = [
        _drop_metrics_and_reliability_signals(result)
        for result in read_all_results()
        if not is_scaling_benchmark(result) and not is_models_scalability_result(result)
    ]

    html = BASE_TEMPLATE.render(
        title="sklbench hardware comparison dashboard",
        rows=[render_comparison(all_results)],
    )

    output = dashboard_output_path("hardware_comparisons.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
