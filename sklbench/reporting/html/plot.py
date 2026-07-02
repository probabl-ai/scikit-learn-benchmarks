from collections import defaultdict
from html import escape
import itertools
import json
import math
from statistics import median

from ..matching import Match
from .templates import PLOT_NOTES_TEMPLATE


PLOTLY_DEFAULT_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


chart_ids = itertools.count()


def _safe_json(value):
    return json.dumps(value, sort_keys=True, default=str).replace("</", "<\\/")


def _format_speedup_tick(value: float) -> str:
    if value >= 1:
        return f"{value:g}x"
    if value >= 0.001:
        return f"{value:.3f}".rstrip("0").rstrip(".") + "x"
    return f"{value:.3g}x"


def _hover_value(value):
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _hover_lines(value, prefix=""):
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines = []
        for key, nested_value in sorted(value.items()):
            if isinstance(nested_value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_hover_lines(nested_value, prefix=f"{prefix}  "))
            elif isinstance(nested_value, list):
                lines.append(f"{prefix}{key}: {json.dumps(nested_value, default=str)}")
            else:
                lines.append(f"{prefix}{key}: {_hover_value(nested_value)}")
        return lines
    return [f"{prefix}{_hover_value(value)}"]


def _custom_format(t):
    if t < 10:
        return f"{t:.2g}ms"
    if t < 1000:
        return f"{round(t)}ms"
    if t < 10_000:
        return f"{t / 1000:.2g}s"
    return f"{round(t / 1000)}s"


def _warning_tooltip_line(warning) -> str:
    message = warning.short_message or ""
    if not message:
        return warning.icon
    return f"{warning.icon} {message}"


def _estimator_name(match: Match) -> str:
    return match.matched_result.case.get("algorithm", {}).get(
        "estimator", "unknown"
    )


def _format_point_count(count: int) -> str:
    return f"{count} point" if count == 1 else f"{count} points"


def _format_estimator_counts(estimator_counts: dict[str, int]) -> str:
    parts = [
        f"{escape(estimator)} ({count})"
        for estimator, count in sorted(
            estimator_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return ", ".join(parts)


def _marker_notes_html(matches: list[Match]) -> str:
    metric_mismatch_count = sum(not match.metrics_match for match in matches)
    warning_counts = {}
    warning_order = []

    for match in matches:
        for warning in match.warnings:
            key = (warning.icon, warning.message)
            if key not in warning_order and match.metrics_match:
                warning_order.append(key)
            if key not in warning_counts:
                warning_counts[key] = {
                    "warning": warning,
                    "estimators": defaultdict(int),
                }
            warning_counts[key]["estimators"][_estimator_name(match)] += 1

    warnings = []
    if warning_order:
        for key in warning_order:
            warning = warning_counts[key]["warning"]
            estimator_counts = warning_counts[key]["estimators"]
            warnings.append(
                {
                    "estimator_counts": _format_estimator_counts(estimator_counts),
                    "icon": warning.icon,
                    "message": warning.message,
                }
            )
    return PLOT_NOTES_TEMPLATE.render(
        metric_mismatch_count=metric_mismatch_count,
        metric_mismatch_label=_format_point_count(metric_mismatch_count),
        warnings=warnings,
    )


def _hover_text(match: Match) -> str:
    result = match.matched_result
    base = match.base_result
    algorithm = result.case.get("algorithm", {})
    data = result.case.get("data", {})
    warning_lines = [_warning_tooltip_line(warning) for warning in match.warnings]
    lines = [
        f"<b>speed-up: {match.speedup:.2g}x</b> "
        f"({_custom_format(median(base.times))} vs {_custom_format(median(result.times))})",
    ]
    metric_differences = match.metrics_differences
    if metric_differences:
        lines.append("<b>metrics differ</b>")
        lines.extend(escape(difference) for difference in metric_differences)
    if warning_lines:
        lines.extend(escape(line) for line in warning_lines)
    estimator_params = "<br>".join(
        escape(line) for line in _hover_lines(algorithm.get("estimator_params", {}))
    )
    data_params = "<br>".join(escape(line) for line in _hover_lines(data))
    lines.extend(
        [
            "<br><b>estimator params</b>",
            estimator_params,
            "<br><b>data params</b>",
            data_params,
        ]
    )
    return "<br>".join(lines)


def _variant_offsets(variants: list[str]) -> dict[str, float]:
    if len(variants) <= 1:
        return {variant: 0 for variant in variants}
    step = 0.14
    center = (len(variants) - 1) / 2
    return {variant: (index - center) * step for index, variant in enumerate(variants)}


def variant_color_map(variants: list[str]) -> dict[str, str]:
    return {
        variant: PLOTLY_DEFAULT_COLORS[index % len(PLOTLY_DEFAULT_COLORS)]
        for index, variant in enumerate(variants)
    }


def _metrics_match(match: Match) -> bool:
    return match.metrics_match


def _marker_symbol(match: Match) -> str:
    if not _metrics_match(match):
        return "diamond-open"
    if match.warnings:
        return "square"
    return "circle"


def _trace_sort_key(match: Match) -> tuple[bool, str]:
    return (_marker_symbol(match) != "circle", _estimator_name(match))


def _trace_variant(match: Match) -> str:
    return match.matched_result.implementation.short_name


def _has_histogram_splits_warning(match: Match) -> bool:
    return any("histogram-based splits" in warning.message for warning in match.warnings)


def _x_variant(match: Match) -> str:
    variant = match.matched_result.implementation.short_name
    if match.matched_result.category != "tree-based":
        return variant

    max_bins_label = (
        "1-max_bins_lt_n_samples"
        if _has_histogram_splits_warning(match)
        else "0-max_bins_eq_n_samples"
    )
    return f"{variant} / {max_bins_label}"


def speedup_plot_html(
    matches: list[Match],
    *,
    variant_colors: dict[str, str] | None = None,
    trace_variant=None,
    x_variant=None,
    variant_sort_key=None,
):
    if not matches:
        return '<div class="empty">No matches for this group.</div>'

    chart_id = f"chart-{next(chart_ids)}"
    if trace_variant is None:
        trace_variant = _trace_variant
    if x_variant is None:
        x_variant = _x_variant
    if variant_sort_key is None:
        variant_sort_key = lambda variant: variant
    if variant_colors is None:
        variant_colors = variant_color_map(
            sorted({trace_variant(match) for match in matches}, key=variant_sort_key)
        )
    estimators = sorted(
        {
            match.matched_result.case.get("algorithm", {}).get("estimator", "unknown")
            for match in matches
        }
    )
    estimator_positions = {
        estimator: index for index, estimator in enumerate(estimators)
    }
    x_variants = sorted({x_variant(match) for match in matches}, key=variant_sort_key)
    offsets = _variant_offsets(x_variants)
    grouped: dict[str, list[Match]] = {}
    for match in matches:
        grouped.setdefault(trace_variant(match), []).append(match)

    traces = []
    for variant, variant_matches in sorted(
        grouped.items(), key=lambda item: variant_sort_key(item[0])
    ):
        variant_matches = sorted(
            variant_matches,
            key=_trace_sort_key,
        )
        marker_symbols = []
        for match in variant_matches:
            marker_symbols.append(_marker_symbol(match))
        marker = {
            "size": 10,
            "symbol": marker_symbols,
        }
        color = variant_colors.get(variant)
        if color is not None:
            marker["color"] = color
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": variant,
                "showlegend": True,
                "x": [
                    estimator_positions[
                        match.matched_result.case.get("algorithm", {}).get(
                            "estimator", "unknown"
                        )
                    ]
                    + offsets[x_variant(match)]
                    for match in variant_matches
                ],
                "y": [math.log2(match.speedup) for match in variant_matches],
                "text": [_hover_text(match) for match in variant_matches],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": marker,
            }
        )

    values = [math.log2(match.speedup) for match in matches]
    min_tick = math.floor(min(min(values), 0))
    max_tick = math.ceil(max(max(values), 0))
    tick_values = list(range(min_tick, max_tick + 1))
    marker_notes = _marker_notes_html(matches)
    layout = {
        "xaxis": {
            "tickmode": "array",
            "tickvals": list(range(len(estimators))),
            "ticktext": estimators,
            "range": [-0.5, len(estimators) - 0.5],
        },
        "yaxis": {
            "title": "speed-up",
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": [_format_speedup_tick(2**tick) for tick in tick_values],
        },
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y",
                "y0": 0,
                "y1": 0,
                "line": {"color": "#666", "width": 1, "dash": "dash"},
            }
        ],
        "margin": {"l": 70, "r": 20, "t": 20, "b": 110},
        "showlegend": True,
        "legend": {"orientation": "h"},
    }
    return f"""<div id="{chart_id}" class="chart"></div>
<script>
  Plotly.newPlot("{chart_id}", {_safe_json(traces)}, {_safe_json(layout)}, {{responsive: true}});
</script>
{marker_notes}"""
