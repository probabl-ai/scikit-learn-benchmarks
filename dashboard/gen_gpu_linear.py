from html import escape
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reporting.envs import read_env, summarize_hardware_env, summarize_software_env
from reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    assemble_plots_in_grid,
    custom_format,
    render_software_tabs,
    variant_color_map,
    _hover_lines,
    _safe_json,
)
from reporting.matching import (
    MatchWarning,
    append_cpu_fallback_warning,
    read_all_results,
    date_range,
    Result,
)


GPU_HARDWARES = {
    "01ba0e": {"name": "NVIDIA L4", "short_name": "L4"},
    "268063": {"name": "Intel Arc B390", "short_name": "B390"},
}
METHODS = ["fit", "predict"]
HARDWARE_ORDER = ["B390", "L4"]
MODEL_ORDER = [
    "LinearRegression",
    "Ridge",
    "RidgeClassifier",
    "LogisticRegression",
]


def is_gpu_result(result: Result) -> bool:
    implementation = result.implementation
    return (
        result.hardware_hash in GPU_HARDWARES
        and result.category == "linear"
        and implementation.short_name != "sklearn"
        and implementation.device in {"cuda", "gpu", "xpu"}
    )


def variant_offsets(variants: list[str]) -> dict[str, float]:
    if len(variants) <= 1:
        return {variant: 0 for variant in variants}
    step = 0.14
    center = (len(variants) - 1) / 2
    return {
        variant: (index - center) * step
        for index, variant in enumerate(variants)
    }


def trace_label(result: Result) -> str:
    hardware = GPU_HARDWARES[result.hardware_hash]
    return f"{result.implementation.short_name}-{hardware['short_name']}"


def trace_sort_key(label: str) -> tuple[int, int, str]:
    implementation, _, hardware = label.rpartition("-")
    implementation_rank = 0 if implementation == "sklearnex-gpu" else 1
    hardware_rank = HARDWARE_ORDER.index(hardware)
    return implementation_rank, hardware_rank, implementation


def result_warnings(result: Result) -> list[MatchWarning]:
    warnings = []
    append_cpu_fallback_warning(result, warnings)
    return warnings


def warning_tooltip_line(warning: MatchWarning) -> str:
    if warning.short_message:
        return f"{warning.icon} {warning.short_message}"
    return warning.icon


def warning_annotation_lines(warnings: list[MatchWarning]) -> list[str]:
    lines = []
    seen = set()
    for warning in warnings:
        key = (warning.icon, warning.message)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{escape(warning.icon)}: {escape(warning.message)}")
    return lines


def result_hover_text(result: Result, hardware_name: str) -> str:
    algorithm = result.case.get("algorithm", {})
    data = result.case.get("data", {})
    warnings = result_warnings(result)
    warning_lines = [warning_tooltip_line(warning) for warning in warnings]
    estimator_params = "<br>".join(
        escape(line) for line in _hover_lines(algorithm.get("estimator_params", {}))
    )
    data_params = "<br>".join(escape(line) for line in _hover_lines(data))
    lines = [
        f"<b>{escape(trace_label(result))}</b>",
        f"{escape(hardware_name)} / {escape(result.method)}",
        f"time: {custom_format(median(result.times))}",
    ]
    if warning_lines:
        lines.extend(escape(line) for line in warning_lines)
    lines.extend(
        [
            "<br><b>estimator params</b>",
            estimator_params,
            "<br><b>data params</b>",
            data_params,
        ]
    )
    return "<br>".join(lines)


def timing_plot_html(
    results: list[Result],
    *,
    method: str,
    trace_colors: dict[str, str],
) -> str:
    plot_results = [
        result
        for result in results
        if result.method == method
    ]
    if not plot_results:
        return '<div class="empty">No GPU benchmark results.</div>'

    chart_id = f"chart-gpu-{method}".replace("-", "_").replace(" ", "_")
    estimators = [
        estimator
        for estimator in MODEL_ORDER
        if any(
            result.case.get("algorithm", {}).get("estimator") == estimator
            for result in plot_results
        )
    ]
    estimator_positions = {
        estimator: index for index, estimator in enumerate(estimators)
    }
    labels = sorted(
        {trace_label(result) for result in plot_results},
        key=trace_sort_key,
    )
    offsets = variant_offsets(labels)

    traces = []
    for label in labels:
        label_results = [
            result
            for result in plot_results
            if trace_label(result) == label
        ]
        label_results = sorted(
            label_results,
            key=lambda result: (
                result.case.get("algorithm", {}).get("estimator", "unknown"),
                median(result.times),
            ),
        )
        marker_symbols = [
            "square" if result_warnings(result) else "circle"
            for result in label_results
        ]
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": label,
                "showlegend": True,
                "x": [
                    estimator_positions[
                        result.case.get("algorithm", {}).get("estimator", "unknown")
                    ]
                    + offsets[label]
                    for result in label_results
                ],
                "y": [median(result.times) for result in label_results],
                "text": [
                    result_hover_text(
                        result, GPU_HARDWARES[result.hardware_hash]["name"]
                    )
                    for result in label_results
                ],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "size": 10,
                    "color": trace_colors[label],
                    "symbol": marker_symbols,
                },
            }
        )

    all_warnings = [
        warning
        for result in plot_results
        for warning in result_warnings(result)
    ]
    annotations = []
    bottom_margin = 110
    warning_lines = warning_annotation_lines(all_warnings)
    if warning_lines:
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.28,
                "xanchor": "left",
                "yanchor": "top",
                "align": "left",
                "showarrow": False,
                "text": "<br>".join(warning_lines),
                "font": {"size": 12, "color": "#5f6368"},
            }
        )
        bottom_margin += 18 * len(warning_lines)

    layout = {
        "xaxis": {
            "tickmode": "array",
            "tickvals": list(range(len(estimators))),
            "ticktext": estimators,
            "range": [-0.5, len(estimators) - 0.5],
        },
        "yaxis": {
            "title": "median time [ms]",
            "type": "log",
        },
        "margin": {"l": 70, "r": 20, "t": 20, "b": bottom_margin},
        "annotations": annotations,
        "showlegend": True,
        "legend": {"orientation": "h"},
    }
    return f"""<div id="{chart_id}" class="chart"></div>
<script>
  Plotly.newPlot("{chart_id}", {_safe_json(traces)}, {_safe_json(layout)}, {{responsive: true}});
</script>"""


def render_hardware_summaries() -> str:
    summaries = [
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hash_)))
        for hash_ in GPU_HARDWARES
    ]
    return f'<section class="summary-grid">{"".join(summaries)}</section>'


def render_software_hardware_tabs(
    results: list[Result], *, trace_colors: dict[str, str]
) -> str:
    elements = []
    seen_labels = set()
    for result in sorted(results, key=lambda result: trace_sort_key(trace_label(result))):
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
    results = [result for result in read_all_results() if is_gpu_result(result)]
    trace_colors = variant_color_map(
        sorted({trace_label(result) for result in results}, key=trace_sort_key)
    )

    plots = []
    for method in METHODS:
        plots.append(
            {
                "method": method,
                "comparison": "GPU",
                "plot": timing_plot_html(
                    results,
                    method=method,
                    trace_colors=trace_colors,
                ),
            }
        )

    html = BASE_TEMPLATE.render(
        title="sklbench GPU linear model dashboard",
        rows=[
            DATE_RANGE_TEMPLATE.render(date_range(results)),
            render_hardware_summaries(),
            render_software_hardware_tabs(results, trace_colors=trace_colors),
            assemble_plots_in_grid(
                plots,
                rows={"method": METHODS},
                columns={"comparison": ["GPU"]},
            ),
        ],
    )

    output = Path("dashboard/gpu_linear.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
