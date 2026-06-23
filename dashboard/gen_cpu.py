from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.output import dashboard_output_path
from reporting.envs import read_env, summarize_hardware_env
from reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    assemble_plots_in_grid,
    render_software_hardware_tabs,
    speedup_plot_html,
    variant_color_map,
)
from reporting.matching import (
    append_iterations_warning,
    date_range,
    find_matches,
    read_all_results,
    Result,
)
from reporting.utils import stable_json, without_keys


HARDWARES = {
    "268063": {"name": "Intel laptop", "short_name": "laptop"},
    "0f5327": {"name": "AMD 48 cores", "short_name": "AMD48"},
    "01ba0e": {"name": "AMD + Nvidia L4 GPU", "short_name": "L4-host"},
}
BASELINE_HARDWARE_HASH = "268063"
BASELINE_LABEL = "sklearn-laptop"
BASE_IMPLEMENTATION = "sklearn"
CATEGORIES = ["linear", "tree-based", "clustering"]
METHODS = ["fit", "predict"]


def is_baseline_result(result: Result) -> bool:
    return (
        result.hardware_hash == BASELINE_HARDWARE_HASH
        and result.implementation.short_name == BASE_IMPLEMENTATION
    )


def is_candidate_result(result: Result) -> bool:
    return (
        result.hardware_hash in HARDWARES
        and result.hardware_hash != BASELINE_HARDWARE_HASH
        and result.implementation.short_name == BASE_IMPLEMENTATION
    )


def trace_label(result: Result) -> str:
    hardware = HARDWARES[result.hardware_hash]
    return f"sklearn-{hardware['short_name']}"


def trace_sort_key(label: str) -> tuple[int, str]:
    order = ["sklearn-AMD48", "sklearn-L4-host"]
    if label in order:
        return order.index(label), label
    return len(order), label


def cpu_match_key(result: Result) -> str:
    case = without_keys(
        result.case,
        excluded_names={"implementation", "max_bins", "n_jobs"},
    )
    case["method"] = result.method
    return stable_json(case)


def result_matches(base_res: Result, candidate: Result) -> tuple[bool, list]:
    assert base_res.implementation.short_name == BASE_IMPLEMENTATION
    assert candidate.implementation.short_name == BASE_IMPLEMENTATION
    assert base_res.hardware_hash == BASELINE_HARDWARE_HASH
    assert candidate.hardware_hash != BASELINE_HARDWARE_HASH

    warnings = []
    append_iterations_warning(base_res, candidate, warnings)

    return cpu_match_key(base_res) == cpu_match_key(candidate), warnings


def match_trace_label(match) -> str:
    return trace_label(match.matched_result)


def render_hardware_summaries() -> str:
    summaries = [
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hash_)))
        for hash_ in HARDWARES
    ]
    return f'<section class="summary-grid">{"".join(summaries)}</section>'


if __name__ == "__main__":
    all_results = read_all_results()
    baseline_results = [
        result for result in all_results if is_baseline_result(result)
    ]
    candidate_results = [
        result for result in all_results if is_candidate_result(result)
    ]
    trace_colors = variant_color_map(
        sorted({trace_label(result) for result in candidate_results}, key=trace_sort_key)
    )

    plots = []
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
                result_matches,
                match_key=cpu_match_key,
            )
            plots.append(
                {
                    "category": category,
                    "method": method,
                    "plot": speedup_plot_html(
                        matches,
                        variant_colors=trace_colors,
                        trace_variant=match_trace_label,
                        x_variant=match_trace_label,
                        variant_sort_key=trace_sort_key,
                    ),
                }
            )

    html = BASE_TEMPLATE.render(
        title="sklbench CPU dashboard",
        rows=[
            DATE_RANGE_TEMPLATE.render(date_range(baseline_results + candidate_results)),
            render_hardware_summaries(),
            render_software_hardware_tabs(
                baseline_results,
                candidate_results,
                baseline_label=BASELINE_LABEL,
                comparison_label=trace_label,
                comparison_sort_key=trace_sort_key,
                variant_colors=trace_colors,
            ),
            assemble_plots_in_grid(
                plots,
                rows={"category": CATEGORIES},
                columns={"method": METHODS},
            ),
        ],
    )

    output = dashboard_output_path("cpu.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
