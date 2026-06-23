from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reporting.envs import read_env, summarize_hardware_env, summarize_software_env
from reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    assemble_plots_in_grid,
    render_software_tabs,
    speedup_plot_html,
    variant_color_map,
)
from reporting.matching import (
    append_cpu_fallback_warning,
    append_iterations_warning,
    find_matches,
    read_all_results,
    date_range,
    Result,
)


GPU_HARDWARES = {
    "01ba0e": {"name": "NVIDIA L4", "short_name": "L4"},
    "268063": {"name": "Intel Arc B390", "short_name": "B390"},
}
BASELINE_HARDWARE_HASH = "268063"
BASELINE_LABEL = "vanilla sklearn"
BASE_IMPLEMENTATION = "sklearn"
METHODS = ["fit", "predict"]
HARDWARE_ORDER = ["B390", "L4"]


def is_gpu_result(result: Result) -> bool:
    implementation = result.implementation
    return (
        result.hardware_hash in GPU_HARDWARES
        and result.category == "linear"
        and implementation.short_name != "sklearn"
        and implementation.device in {"cuda", "gpu", "xpu"}
    )


def is_baseline_result(result: Result) -> bool:
    return (
        result.hardware_hash == BASELINE_HARDWARE_HASH
        and result.category == "linear"
        and result.implementation.short_name == BASE_IMPLEMENTATION
    )


def trace_label(result: Result) -> str:
    hardware = GPU_HARDWARES[result.hardware_hash]
    return f"{result.implementation.short_name}-{hardware['short_name']}"


def trace_sort_key(label: str) -> tuple[int, int, str]:
    implementation, _, hardware = label.rpartition("-")
    implementation_rank = 0 if implementation == "sklearnex-gpu" else 1
    hardware_rank = HARDWARE_ORDER.index(hardware)
    return implementation_rank, hardware_rank, implementation


def result_matches(base_res: Result, candidate: Result) -> tuple[bool, list]:
    assert base_res.implementation.short_name == BASE_IMPLEMENTATION
    assert candidate.implementation.short_name != BASE_IMPLEMENTATION

    warnings = []
    append_iterations_warning(base_res, candidate, warnings)
    append_cpu_fallback_warning(candidate, warnings)

    return base_res.minimal_match_key == candidate.minimal_match_key, warnings


def match_trace_label(match) -> str:
    return trace_label(match.matched_result)


def render_hardware_summaries() -> str:
    summaries = [
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hash_)))
        for hash_ in GPU_HARDWARES
    ]
    return f'<section class="summary-grid">{"".join(summaries)}</section>'


def render_software_hardware_tabs(
    baseline_results: list[Result],
    gpu_results: list[Result],
    *,
    trace_colors: dict[str, str],
) -> str:
    elements = []
    if baseline_results:
        baseline = baseline_results[0]
        software_summary = summarize_software_env(
            read_env("software", baseline.software_hash),
            baseline.implementation,
        )
        software_summary["name"] = BASELINE_LABEL
        hardware_summary = summarize_hardware_env(
            read_env("hardware", baseline.hardware_hash)
        )
        elements.append(
            SOFTWARE_TEMPLATE.render(**software_summary)
            + HARDWARE_TEMPLATE.render(hardware_summary)
        )

    seen_labels = set()
    for result in sorted(
        gpu_results, key=lambda result: trace_sort_key(trace_label(result))
    ):
        label = trace_label(result)
        if label in seen_labels:
            continue
        seen_labels.add(label)

        software_summary = summarize_software_env(
            read_env("software", result.software_hash),
            result.implementation,
        )
        software_summary["name"] = label
        hardware_summary = summarize_hardware_env(
            read_env("hardware", result.hardware_hash)
        )
        elements.append(
            SOFTWARE_TEMPLATE.render(**software_summary)
            + HARDWARE_TEMPLATE.render(hardware_summary)
        )
    return render_software_tabs(elements, variant_colors=trace_colors)


if __name__ == "__main__":
    all_results = read_all_results()
    baseline_results = [
        result for result in all_results if is_baseline_result(result)
    ]
    gpu_results = [result for result in all_results if is_gpu_result(result)]
    trace_colors = variant_color_map(
        sorted({trace_label(result) for result in gpu_results}, key=trace_sort_key)
    )

    plots = []
    for method in METHODS:
        matches = find_matches(
            [result for result in baseline_results if result.method == method],
            [result for result in gpu_results if result.method == method],
            result_matches,
        )
        plots.append(
            {
                "method": method,
                "comparison": f"speed-up vs {BASELINE_LABEL}",
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
        title="sklbench GPU linear model dashboard",
        rows=[
            DATE_RANGE_TEMPLATE.render(date_range(baseline_results + gpu_results)),
            render_hardware_summaries(),
            render_software_hardware_tabs(
                baseline_results,
                gpu_results,
                trace_colors=trace_colors,
            ),
            assemble_plots_in_grid(
                plots,
                rows={"method": METHODS},
                columns={"comparison": [f"speed-up vs {BASELINE_LABEL}"]},
            ),
        ],
    )

    output = Path("dashboard/gpu_linear.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
