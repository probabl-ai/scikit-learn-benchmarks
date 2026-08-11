from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    assemble_plots_in_grid,
    detailed_results_table_html,
    render_hardware_tabs,
    render_software_hardware_tabs,
    speedup_plot_html,
    variant_color_map,
)
from sklbench.reporting.envs import is_vanilla_sklearn
from sklbench.reporting.matching import (
    append_cpu_fallback_warning,
    append_iterations_warning,
    append_max_bins_warning,
    find_matches,
    read_all_results,
    read_failed_records,
    date_range,
    BenchmarkRecord,
    MethodResult,
)
from sklbench.reporting.utils import groupby, stable_json, without_keys


GPU_HARDWARES = {
    "4cb66b": {"name": "NVIDIA L4", "short_name": "L4"},
    "3b5e61": {"name": "Intel Arc B390", "short_name": "B390"},
}
CPU_HARDWARES = {
    "3b5e61": {"name": "Intel laptop", "short_name": "laptop"},
    "534824": {"name": "Intel GNR 172 CPU cores", "short_name": "GNR"},
}
CPU_BASELINE_HARDWARE_HASH = "3b5e61"
BASELINE_LABEL = "vanilla sklearn"
BASE_IMPLEMENTATION = "sklearn"
CATEGORIES = ["linear", "tree-based", "clustering"]
METHODS = ["fit", "predict"]
GPU_HARDWARE_ORDER = ["B390", "L4"]
CPU_HARDWARE_ORDER = ["laptop", "GNR"]


def is_alt_sklearn_build(result: MethodResult | BenchmarkRecord) -> bool:
    return (
        result.implementation.short_name == BASE_IMPLEMENTATION
        and not is_vanilla_sklearn(result.software_hash)
    )


def _hardware_match_key(result: MethodResult) -> str:
    case = without_keys(
        result.case,
        excluded_names={"implementation", "max_bins", "n_jobs"},
    )
    case["method"] = result.method
    return stable_json(case)


def _hardware_table_comparison_key(result: MethodResult) -> str:
    return stable_json(
        without_keys(
            result.case,
            excluded_names={"implementation", "max_bins", "n_jobs"},
        )
    )


def _is_gpu_result(result: MethodResult | BenchmarkRecord, *, hardware_hashes: set[str]) -> bool:
    implementation = result.implementation
    return (
        result.hardware_hash in hardware_hashes
        and result.category == "linear"
        and implementation.short_name != "sklearn"
        and implementation.device in {"cuda", "gpu", "xpu"}
    )


def is_linear_baseline_result(result: MethodResult | BenchmarkRecord) -> bool:
    return (
        result.hardware_hash == CPU_BASELINE_HARDWARE_HASH
        and result.category == "linear"
        and result.implementation.short_name == BASE_IMPLEMENTATION
    )


def is_gpu_candidate_result(result: MethodResult | BenchmarkRecord) -> bool:
    return _is_gpu_result(result, hardware_hashes=set(GPU_HARDWARES))


def gpu_trace_label(result: MethodResult) -> str:
    hardware = GPU_HARDWARES[result.hardware_hash]
    return f"{result.implementation.short_name}-{hardware['short_name']}"


def linear_trace_label(result: MethodResult) -> str:
    if result.implementation.device in {None, "default", "cpu"}:
        return cpu_trace_label(result)
    return gpu_trace_label(result)


def linear_trace_sort_key(label: str) -> tuple[int, int, int, str]:
    implementation, _, hardware = label.rpartition("-")
    if hardware in CPU_HARDWARE_ORDER:
        hardware_rank, implementation_rank, _ = cpu_trace_sort_key(label)
        return 0, hardware_rank, implementation_rank, label

    implementation_rank = 0 if implementation == "sklearnex-gpu" else 1
    hardware_rank = GPU_HARDWARE_ORDER.index(hardware)
    return 1, implementation_rank, hardware_rank, implementation


def linear_result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list]:
    assert base_res.implementation.short_name == BASE_IMPLEMENTATION

    warnings = []
    if candidate.is_sklearnex_tree:
        append_max_bins_warning(base_res, candidate, warnings)
    append_iterations_warning(base_res, candidate, warnings)
    append_cpu_fallback_warning(candidate, warnings)

    return _hardware_match_key(base_res) == _hardware_match_key(candidate), warnings


def linear_match_trace_label(match) -> str:
    return linear_trace_label(match.matched_result)


def is_cpu_baseline_result(result: MethodResult | BenchmarkRecord) -> bool:
    return (
        result.hardware_hash == CPU_BASELINE_HARDWARE_HASH
        and result.implementation.short_name == BASE_IMPLEMENTATION
    )


def is_cpu_candidate_result(result: MethodResult | BenchmarkRecord) -> bool:
    return (
        result.hardware_hash in CPU_HARDWARES
        and not (
            result.hardware_hash == CPU_BASELINE_HARDWARE_HASH
            and result.implementation.short_name == BASE_IMPLEMENTATION
        )
        and result.implementation.short_name in {BASE_IMPLEMENTATION, "sklearnex-cpu"}
    )


def cpu_trace_label(result: MethodResult) -> str:
    hardware = CPU_HARDWARES[result.hardware_hash]
    return f"{result.implementation.short_name}-{hardware['short_name']}"


def cpu_trace_sort_key(label: str) -> tuple[int, int, str]:
    implementation, _, hardware = label.rpartition("-")
    hardware_rank = CPU_HARDWARE_ORDER.index(hardware)
    implementation_order = [BASE_IMPLEMENTATION, "sklearnex-cpu"]
    if implementation in implementation_order:
        return hardware_rank, implementation_order.index(implementation), label
    return hardware_rank, len(implementation_order), label


def cpu_result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list]:
    assert base_res.implementation.short_name == BASE_IMPLEMENTATION
    assert candidate.implementation.short_name in {BASE_IMPLEMENTATION, "sklearnex-cpu"}
    assert base_res.hardware_hash == CPU_BASELINE_HARDWARE_HASH

    warnings = []
    if candidate.is_sklearnex_tree:
        append_max_bins_warning(base_res, candidate, warnings)
    append_iterations_warning(base_res, candidate, warnings)

    return _hardware_match_key(base_res) == _hardware_match_key(candidate), warnings


def cpu_match_trace_label(match) -> str:
    return cpu_trace_label(match.matched_result)


def _comparison_page(rows: list[str]) -> str:
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


def render_linear_comparison(
    all_results: list[MethodResult],
    failed_records: list[BenchmarkRecord],
    *,
    is_candidate_result,
) -> str:
    baseline_results = [
        result for result in all_results if is_linear_baseline_result(result)
    ]
    candidate_results = [
        result for result in all_results if is_candidate_result(result)
    ]
    # Non-baseline (candidate) failures: shown on the plots too, at the bottom
    # of their model-variant column. A failed record carries no fit/predict
    # method info, so it's shown on both method plots.
    candidate_failed_records = [
        record for record in failed_records if is_candidate_result(record)
    ]
    trace_colors = variant_color_map(
        sorted(
            {linear_trace_label(result) for result in candidate_results},
            key=linear_trace_sort_key,
        )
    )

    plots = []
    matches_by_method = {}
    for method in METHODS:
        matches = find_matches(
            [result for result in baseline_results if result.method == method],
            [result for result in candidate_results if result.method == method],
            linear_result_matches,
            match_key=_hardware_match_key,
        )
        matches_by_method[method] = matches
        plots.append(
            {
                "method": method,
                "comparison": f"speed-up vs {BASELINE_LABEL}",
                "point_count": len(matches),
                "plot": speedup_plot_html(
                    matches,
                    variant_colors=trace_colors,
                    trace_variant=linear_match_trace_label,
                    x_variant=linear_match_trace_label,
                    variant_sort_key=linear_trace_sort_key,
                    failed_records=candidate_failed_records,
                ),
            }
        )

    return _comparison_page(
        [
            DATE_RANGE_TEMPLATE.render(
                date_range(baseline_results + candidate_results)
            ),
            render_software_hardware_tabs(
                baseline_results,
                candidate_results,
                baseline_label=BASELINE_LABEL,
                comparison_label=linear_trace_label,
                comparison_sort_key=linear_trace_sort_key,
                variant_colors=trace_colors,
            ),
            assemble_plots_in_grid(
                plots,
                rows={"method": METHODS},
                columns={"comparison": [f"speed-up vs {BASELINE_LABEL}"]},
                details_after_grid=[
                    detailed_results_table_html(
                        "linear",
                        matches_by_method,
                        baseline_label=BASELINE_LABEL,
                        variant_label=linear_trace_label,
                        comparison_key=_hardware_table_comparison_key,
                    )
                ],
            ),
        ]
    )


def render_cpu_comparison(
    all_results: list[MethodResult], failed_records: list[BenchmarkRecord]
) -> str:
    baseline_results = [
        result for result in all_results if is_cpu_baseline_result(result)
    ]
    candidate_results = [
        result for result in all_results if is_cpu_candidate_result(result)
    ]
    # Non-baseline (candidate) failures: shown on the plots too, at the bottom
    # of their model-variant column. A failed record carries no fit/predict
    # method info, so it's shown on both method plots.
    candidate_failed_by_category = groupby(
        [record for record in failed_records if is_cpu_candidate_result(record)],
        lambda record: record.category,
    )
    trace_colors = variant_color_map(
        sorted(
            {cpu_trace_label(result) for result in candidate_results},
            key=cpu_trace_sort_key,
        )
    )

    plots = []
    matches_by_category = {}
    for category in CATEGORIES:
        for method in METHODS:
            matches = find_matches(
                [
                    result
                    for result in baseline_results
                    if result.category == category and result.method == method
                ],
                [
                    result
                    for result in candidate_results
                    if result.category == category and result.method == method
                ],
                cpu_result_matches,
                match_key=_hardware_match_key,
            )
            matches_by_category.setdefault(category, {})[method] = matches
            plots.append(
                {
                    "category": category,
                    "method": method,
                    "point_count": len(matches),
                    "plot": speedup_plot_html(
                        matches,
                        variant_colors=trace_colors,
                        trace_variant=cpu_match_trace_label,
                        x_variant=cpu_match_trace_label,
                        variant_sort_key=cpu_trace_sort_key,
                        failed_records=candidate_failed_by_category.get(category, []),
                    ),
                }
            )

    return _comparison_page(
        [
            DATE_RANGE_TEMPLATE.render(
                date_range(baseline_results + candidate_results)
            ),
            render_software_hardware_tabs(
                baseline_results,
                candidate_results,
                baseline_label=BASELINE_LABEL,
                comparison_label=cpu_trace_label,
                comparison_sort_key=cpu_trace_sort_key,
                variant_colors=trace_colors,
            ),
            assemble_plots_in_grid(
                plots,
                rows={"category": CATEGORIES},
                columns={"method": METHODS},
                details_by_row={
                    category: detailed_results_table_html(
                        category,
                        matches_by_method,
                        baseline_label=BASELINE_LABEL,
                        variant_label=cpu_trace_label,
                        comparison_key=_hardware_table_comparison_key,
                    )
                    for category, matches_by_method in matches_by_category.items()
                },
            ),
        ]
    )


if __name__ == "__main__":
    all_results = [
        result for result in read_all_results() if not is_alt_sklearn_build(result)
    ]
    failed_records = [
        record for record in read_failed_records() if not is_alt_sklearn_build(record)
    ]
    html = BASE_TEMPLATE.render(
        title="sklbench hardware comparison dashboard",
        rows=[
            render_hardware_tabs(
                [
                    (
                        "Compare GPUs",
                        render_linear_comparison(
                            all_results,
                            failed_records,
                            is_candidate_result=is_gpu_candidate_result,
                        ),
                    ),
                    (
                        "Compare CPUs",
                        render_cpu_comparison(all_results, failed_records),
                    ),
                ]
            ),
        ],
    )

    output = dashboard_output_path("hardware_comparisons.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
