from collections import defaultdict
from dataclasses import dataclass, field
from html import escape
import itertools
import json
import math
from statistics import median

from plotly import graph_objects as go

from ..matching import BenchmarkRecord, Match
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

FALLBACK_COLOR = "#9e9e9e"
FAILED_MARKER_GAP = 0.4  # log2(speed-up) units below the slowest point of a column


@dataclass(frozen=True)
class _FailedMatch:
    """Adapts a failed `BenchmarkRecord` to the `Match`-shaped interface that
    `trace_variant`/`x_variant` callbacks expect (they only ever read
    `.matched_result` and `.warnings`), so failed non-baseline runs can reuse
    the exact same column-placement logic as real matches."""

    matched_result: BenchmarkRecord
    warnings: list = field(default_factory=list)


chart_ids = itertools.count()


def _format_speedup_tick(value: float) -> str:
    if value >= 1:
        return f"{value:g}x"
    denom = round(1 / value)
    if math.isclose(value * denom, 1, rel_tol=1e-9):
        return f"1/{denom}x"
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


GENERATION_KWARGS_HOVER_KEYS = ["n_samples", "n_features", "columns"]


def _generation_kwargs_hover_lines(value, prefix):
    lines = []
    for key in GENERATION_KWARGS_HOVER_KEYS:
        if key not in value:
            continue
        nested_value = value[key]
        if isinstance(nested_value, list):
            lines.append(f"{prefix}{key}: {json.dumps(nested_value, default=str)}")
        else:
            lines.append(f"{prefix}{key}: {_hover_value(nested_value)}")
    return lines


def _hover_lines(value, prefix=""):
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines = []
        for key, nested_value in sorted(value.items()):
            if key == "generation_kwargs" and isinstance(nested_value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(
                    _generation_kwargs_hover_lines(nested_value, f"{prefix}  ")
                )
            elif isinstance(nested_value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_hover_lines(nested_value, prefix=f"{prefix}  "))
            elif isinstance(nested_value, list):
                lines.append(f"{prefix}{key}: {json.dumps(nested_value, default=str)}")
            else:
                lines.append(f"{prefix}{key}: {_hover_value(nested_value)}")
        return lines
    return [f"{prefix}{_hover_value(value)}"]


def format_duration_ms(t):
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


# Shortens tree-estimator names on x-axis tick labels only - everywhere else
# (hover text, warning tooltips, matching) keeps the full estimator name.
_ESTIMATOR_LABEL_ABBREVIATIONS = {
    "HistGradientBoosting": "HGB",
    "RandomForest": "RF",
    "Regressor": "Reg",
    "Classifier": "Clf",
}


def _abbreviate_estimator_label(name: str) -> str:
    for full, short in _ESTIMATOR_LABEL_ABBREVIATIONS.items():
        name = name.replace(full, short)
    return name


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


def _marker_notes_html(matches: list[Match], failed_count: int = 0) -> str:
    metric_mismatch_count = sum(not match.metrics_match for match in matches)
    fallback_count = sum(
        match.matched_result.is_sklearnex_fallback for match in matches
    )
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
        fallback_count=fallback_count,
        fallback_label=_format_point_count(fallback_count),
        failed_count=failed_count,
        failed_label=_format_point_count(failed_count),
        warnings=warnings,
    )


def _format_metric_difference(difference) -> str:
    return (
        f"{difference.metric_name}: {difference.base_repr} (base) "
        f"vs {difference.target_repr} (variant)"
    )


_METHOD_LABELS = {"fit": "train", "predict": "test"}


def _method_label(method: str) -> str:
    return _METHOD_LABELS.get(method, method)


def _metrics_differ_lines(match: Match) -> list[str]:
    """Render metric mismatches, prioritizing predict over fit: predict
    differences (if any) are shown in full, and fit differences are only
    detailed when predict metrics otherwise match - else they're collapsed
    into a one-line note, since predict.metrics_differences is the one users
    read first per match."""
    differences = match.metrics_differences
    if not differences:
        return []

    fit_differences = [d for d in differences if d.method == "fit"]
    other_differences = [d for d in differences if d.method != "fit"]

    if other_differences:
        method = other_differences[0].method
        lines = [f"<b>{escape(_method_label(method))} metrics differ:</b>"]
        lines.extend(escape(_format_metric_difference(d)) for d in other_differences)
        if fit_differences:
            lines.append(f"<i>{escape(_method_label('fit'))} metrics also mismatch</i>")
        return lines

    lines = [f"<b>{escape(_method_label('fit'))} metrics differ:</b>"]
    lines.extend(escape(_format_metric_difference(d)) for d in fit_differences)
    return lines


def _real_dataset_dimensions_line(data: dict, data_desc: dict | None) -> str | None:
    """Real datasets don't carry n_samples/n_features in the case (those are
    only in `generation_kwargs`, which synthetic datasets have), so surface
    the shape recorded in `data_desc` instead."""
    if data.get("generation_kwargs") or not data_desc:
        return None
    samples, features = data_desc.get("samples"), data_desc.get("features")
    if samples is None or features is None:
        return None
    return f"dimensions: {samples:,} x {features}"


def _record_fit_data_desc(record: BenchmarkRecord) -> dict | None:
    for run in record.runs:
        fit_desc = (run.get("data_desc") or {}).get("fit")
        if fit_desc:
            return fit_desc
    return None


def _data_hover_lines(data: dict, data_desc: dict | None) -> list[str]:
    lines = _hover_lines(data)
    dimensions_line = _real_dataset_dimensions_line(data, data_desc)
    if dimensions_line:
        lines.append(dimensions_line)
    return lines


def _hover_text(match: Match) -> str:
    result = match.matched_result
    base = match.base_result
    algorithm = result.case.get("algorithm", {})
    data = result.case.get("data", {})
    warning_lines = [_warning_tooltip_line(warning) for warning in match.warnings]
    lines = [
        f"<b>speed-up: {match.speedup:.2g}x</b> "
        f"({format_duration_ms(median(base.times))} vs {format_duration_ms(median(result.times))})",
    ]
    lines.extend(_metrics_differ_lines(match))
    if result.is_sklearnex_fallback:
        lines.append("<i>fell back to scikit-learn</i>")
    if warning_lines:
        lines.extend(escape(line) for line in warning_lines)
    estimator_params = "<br>".join(
        escape(line) for line in _hover_lines(algorithm.get("estimator_params", {}))
    )
    data_params = "<br>".join(
        escape(line) for line in _data_hover_lines(data, result.data_desc)
    )
    lines.extend(
        [
            "<br><b>estimator params</b>",
            estimator_params,
            "<br><b>data params</b>",
            data_params,
        ]
    )
    return "<br>".join(lines)


def _failed_hover_text(record: BenchmarkRecord) -> str:
    algorithm = record.case.get("algorithm", {})
    data = record.case.get("data", {})
    estimator_params = "<br>".join(
        escape(line) for line in _hover_lines(algorithm.get("estimator_params", {}))
    )
    data_params = "<br>".join(
        escape(line)
        for line in _data_hover_lines(data, _record_fit_data_desc(record))
    )
    return "<br>".join(
        [
            "<b>benchmark run failed</b>",
            "<br><b>estimator params</b>",
            estimator_params,
            "<br><b>data params</b>",
            data_params,
        ]
    )


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


def phase_breakdown_plot_html(
    points: list[dict],
    *,
    phase_order: list[str],
    phase_colors: dict[str, str],
    phase_labels: dict[str, str] | None = None,
    x_title: str = "threads",
) -> str:
    """Stacked bar of phase timings (ms) vs. an x-axis category (e.g. thread
    count), one bar per `points` entry. Each point is
    `{"x": ..., "phases": {phase_name: ms}, "total_ms": ...}`, with an
    optional `"x_label"` overriding `str(x)` as the tick label (e.g. to show
    an actual-vs-requested thread count as `"4 (3)"`) while `x` itself still
    drives sort order. Legend is disabled on every trace - callers render one
    shared legend across a grid of these (see dashboards/gen_hgb_scaling.py)
    rather than repeating it per small multiple."""
    if phase_labels is None:
        phase_labels = {phase: phase for phase in phase_order}
    chart_id = f"phase-breakdown-{next(chart_ids)}"
    points = sorted(points, key=lambda point: point["x"])
    x_values = [str(point.get("x_label", point["x"])) for point in points]

    fig = go.Figure()
    for phase in phase_order:
        y_values = [point["phases"].get(phase, 0.0) for point in points]
        totals = [point["total_ms"] for point in points]
        fig.add_trace(
            go.Bar(
                name=phase_labels[phase],
                x=x_values,
                y=y_values,
                marker={"color": phase_colors[phase]},
                showlegend=False,
                # Duration is pre-formatted in Python (hovertemplate's own
                # %{y:.3g} can't apply format_duration_ms's ms/s unit
                # switch), so it rides along in customdata next to the share.
                customdata=[
                    (format_duration_ms(y), _phase_share(y, total))
                    for y, total in zip(y_values, totals)
                ],
                hovertemplate=(
                    f"{phase_labels[phase]}: "
                    "%{customdata[0]} (%{customdata[1]:.0%})<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis={"type": "category", "title": x_title},
        yaxis={"title": "time (ms)", "rangemode": "tozero"},
        margin={"l": 60, "r": 15, "t": 15, "b": 44},
        showlegend=False,
        template="none",
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="320px",
        div_id=chart_id,
    )


def scaling_line_plot_html(
    series: dict[str, list[tuple[float, float]]],
    *,
    colors: dict[str, str] | None = None,
    x_title: str = "threads",
    y_title: str = "fit time (ms)",
    y_unit: str = "ms",
    x_log: bool = False,
    y_log: bool = False,
    reference_lines: dict[str, list[tuple[float, float]]] | None = None,
) -> str:
    """Simple line plot of `y_title` vs `x_title`, one line per series key
    (e.g. software build). `series` maps a label to a list of (x, y) points.
    Lighter-weight alternative to `phase_breakdown_plot_html` for comparing
    overall scaling behavior across builds, rather than a per-build phase
    breakdown. `y_unit` only affects the hover suffix - callers passing
    non-millisecond `y` values (e.g. gen_models_scalability.py's seconds)
    must override it to match, since it isn't derived from `y_title`.

    `x_log` switches the x-axis to log scale - the intended use is a
    power-of-two sweep (e.g. core counts), so ticks are pinned to powers of
    two spanning `series`' x range (1, 2, 4, ...) rather than Plotly's
    default power-of-10 log ticks, which wouldn't land on them - including
    when the actual max x isn't itself a power of two (e.g. a core count
    capped by the machine's physical core count). `y_log` switches the
    y-axis to log scale - `rangemode: tozero` is skipped in that case, since
    a log axis can't include 0.

    Each point is normally an `(x, y)` pair; a point may add a third element
    - one extra line of hover text (e.g. `"n_iter: 431"`) shown below the x/y
    line, or `""`/`None` for no extra line - for callers that want per-point
    context an aggregate x/y trend can't convey.

    `reference_lines` draws additional dashed, marker-less, grey lines (e.g.
    an ideal-scaling reference) in the same `{label: [(x, y), ...]}` shape as
    `series`, kept visually distinct from the real data traces."""
    chart_id = f"scaling-line-{next(chart_ids)}"
    fig = go.Figure()
    all_x_values = set()
    for label, points in sorted(series.items()):
        points = sorted(points, key=lambda point: point[0])
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        hover_extra = [
            f"<br>{point[2]}" if len(point) > 2 and point[2] else "" for point in points
        ]
        all_x_values.update(x_values)
        color = (colors or {}).get(label)
        fig.add_trace(
            go.Scatter(
                name=label,
                x=x_values,
                y=y_values,
                mode="lines+markers",
                line={"color": color} if color else {},
                marker={"color": color} if color else {},
                customdata=hover_extra,
                hovertemplate=(
                    f"{label}<br>%{{x}} {x_title}: %{{y:.3g}}{y_unit}"
                    "%{customdata}<extra></extra>"
                ),
            )
        )
    for label, points in sorted((reference_lines or {}).items()):
        points = sorted(points, key=lambda point: point[0])
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        all_x_values.update(x_values)
        fig.add_trace(
            go.Scatter(
                name=label,
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": "#999", "dash": "dash", "width": 1},
                hoverinfo="skip",
            )
        )
    xaxis = {"title": x_title}
    if x_log and all_x_values:
        min_x, max_x = min(all_x_values), max(all_x_values)
        start_exp = math.floor(math.log2(max(min_x, 1)))
        end_exp = math.ceil(math.log2(max(max_x, 1)))
        tick_values = [2**exp for exp in range(start_exp, end_exp + 1)]
        xaxis |= {
            "type": "log",
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": [str(x) for x in tick_values],
        }
    yaxis = {"title": y_title, "type": "log"} if y_log else {
        "title": y_title,
        "rangemode": "tozero",
    }
    fig.update_layout(
        xaxis=xaxis,
        yaxis=yaxis,
        margin={"l": 60, "r": 15, "t": 15, "b": 44},
        legend={"orientation": "h", "y": -0.25},
        template="none",
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="340px",
        div_id=chart_id,
    )


def phase_variant_speedup_plot_html(
    points: list[dict],
    *,
    phase_order: list[str],
    phase_labels: dict[str, str] | None = None,
    variant_colors: dict[str, str] | None = None,
    y_title: str = "normalized speed-up",
    secondary_axis_phase: str | None = None,
    secondary_axis_title: str = "relative speed-up",
) -> str:
    """Per-phase speed-up of one or more variant builds vs a baseline, for a
    single thread count. Each of `points` is
    `{"phase": ..., "variant": ..., "y": ..., "hover": "<pre-rendered html>"}`
    - many points typically share the same (phase, variant) x position (one
    per dataset/workload), rendered on top of each other with hover
    distinguishing them, same convention as `speedup_plot_html`.

    x-axis is phase (categorical, ordered/labeled by `phase_order`/
    `phase_labels`), with each phase column split into per-variant
    sub-positions via `_variant_offsets` so more than a single
    baseline-vs-one-variant comparison reads at a glance; color also encodes
    variant. y-axis is linear with a dashed reference line at 0 (no change).

    `secondary_axis_phase`, if given, is a phase key whose points are read
    off a second y-axis on the right (titled `secondary_axis_title`, with
    its own 1x reference line) instead of the shared left one - the caller
    is expected to have put a different quantity in `y` for those points
    (e.g. a plain ratio rather than the normalized share the rest of the
    phases use), since the two axes have unrelated scales. Those points may
    also carry an `"error"` key (same units as `y`) rendered as a Plotly
    error bar, for callers that want to surface measurement noise on the
    secondary quantity rather than just its point estimate."""
    if not points:
        return '<div class="empty">No points for this thread count.</div>'
    if phase_labels is None:
        phase_labels = {phase: phase for phase in phase_order}
    chart_id = f"phase-variant-speedup-{next(chart_ids)}"

    present_phases = [phase for phase in phase_order if any(p["phase"] == phase for p in points)]
    phase_positions = {phase: index for index, phase in enumerate(present_phases)}

    variants = sorted({p["variant"] for p in points})
    offsets = _variant_offsets(variants)
    if variant_colors is None:
        variant_colors = variant_color_map(variants)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        grouped[point["variant"]].append(point)

    fig = go.Figure()
    for variant in variants:
        color = variant_colors.get(variant)
        primary_points = [
            p for p in grouped[variant] if p["phase"] != secondary_axis_phase
        ]
        secondary_points = [
            p for p in grouped[variant] if p["phase"] == secondary_axis_phase
        ]
        fig.add_trace(
            go.Scatter(
                mode="markers",
                name=variant,
                legendgroup=variant,
                x=[
                    phase_positions[p["phase"]] + offsets[variant]
                    for p in primary_points
                ],
                y=[p["y"] for p in primary_points],
                text=[p["hover"] for p in primary_points],
                hovertemplate="%{text}<extra></extra>",
                marker={"size": 9, "color": color},
            )
        )
        if secondary_points:
            # `error` is an optional per-point uncertainty (same units as
            # `y`) - e.g. repeat-to-repeat noise on the ratio itself, as
            # opposed to `y`'s own point estimate - rendered as a Plotly
            # error bar rather than folded into the marker position.
            errors = [p.get("error") for p in secondary_points]
            error_y = (
                {
                    "type": "data",
                    "array": [error or 0.0 for error in errors],
                    "visible": True,
                    "color": color,
                    "thickness": 1,
                    "width": 3,
                }
                if any(error for error in errors)
                else None
            )
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    name=variant,
                    legendgroup=variant,
                    showlegend=False,
                    yaxis="y2",
                    x=[
                        phase_positions[p["phase"]] + offsets[variant]
                        for p in secondary_points
                    ],
                    y=[p["y"] for p in secondary_points],
                    error_y=error_y,
                    text=[p["hover"] for p in secondary_points],
                    hovertemplate="%{text}<extra></extra>",
                    marker={"size": 9, "color": color, "symbol": "diamond"},
                )
            )

    shapes = [
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
    ]
    layout_yaxis = {"title": y_title, "zeroline": True}
    layout_yaxis2 = {}
    if secondary_axis_phase in phase_positions:
        secondary_x = phase_positions[secondary_axis_phase]
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "x0": secondary_x - 0.5,
                "x1": secondary_x + 0.5,
                "yref": "y2",
                "y0": 1,
                "y1": 1,
                "line": {"color": "#666", "width": 1, "dash": "dash"},
            }
        )
        # Both axes are forced symmetric around their own "no change" value
        # (0 here, 1 on the secondary axis below) with the same padding
        # factor, so those two reference points land at the exact same
        # height - left uncoordinated, Plotly autoranges each axis from its
        # own data and the two dashed reference lines end up at unrelated
        # heights, which reads as broken rather than as two views of the
        # same "no change" baseline.
        primary_span = max(
            (abs(p["y"]) for p in points if p["phase"] != secondary_axis_phase),
            default=0.0,
        ) * 1.15 or 1.0
        secondary_span = max(
            (
                abs(p["y"] - 1) + (p.get("error") or 0.0)
                for p in points
                if p["phase"] == secondary_axis_phase
            ),
            default=0.0,
        ) * 1.15 or 0.1
        layout_yaxis["range"] = [-primary_span, primary_span]
        layout_yaxis2 = {
            "yaxis2": {
                "title": {"text": secondary_axis_title, "standoff": 15},
                "overlaying": "y",
                "side": "right",
                "range": [1 - secondary_span, 1 + secondary_span],
            }
        }

    # The secondary axis's title sits further out than its tick labels -
    # the default 20px right margin is enough for tick labels alone but
    # clips/overlaps the rotated title text once a second axis is added.
    right_margin = 70 if layout_yaxis2 else 20

    fig.update_layout(
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(len(present_phases))),
            "ticktext": [phase_labels[phase] for phase in present_phases],
            "range": [-0.5, len(present_phases) - 0.5],
        },
        yaxis=layout_yaxis,
        **layout_yaxis2,
        shapes=shapes,
        margin={"l": 70, "r": right_margin, "t": 20, "b": 90},
        legend={"orientation": "h", "y": -0.2},
        template="none",
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="480px",
        div_id=chart_id,
    )


def _phase_share(value: float, total: float) -> float:
    return value / total if total else 0.0


def _metrics_match(match: Match) -> bool:
    return match.metrics_match


def _marker_symbol(match: Match) -> str:
    if not _metrics_match(match):
        return "diamond-open"
    if match.warnings:
        return "square"
    return "circle"


def _marker_color(match: Match, variant_color: str | None) -> str | None:
    if match.matched_result.is_sklearnex_fallback:
        return FALLBACK_COLOR
    return variant_color


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
    baseline_label: str,
    variant_colors: dict[str, str] | None = None,
    trace_variant=None,
    x_variant=None,
    variant_sort_key=None,
    failed_records: list[BenchmarkRecord] = (),
):
    if not matches and not failed_records:
        return '<div class="empty">No matches for this group.</div>'

    chart_id = f"chart-{next(chart_ids)}"
    if trace_variant is None:
        trace_variant = _trace_variant
    if x_variant is None:
        x_variant = _x_variant
    if variant_sort_key is None:
        variant_sort_key = lambda variant: variant
    failed_matches = [_FailedMatch(record) for record in failed_records]

    estimators = sorted(
        {_estimator_name(match) for match in matches}
        | {_estimator_name(failed) for failed in failed_matches}
    )
    estimator_positions = {
        estimator: index for index, estimator in enumerate(estimators)
    }
    x_variants = sorted(
        {x_variant(match) for match in matches}
        | {x_variant(failed) for failed in failed_matches},
        key=variant_sort_key,
    )
    offsets = _variant_offsets(x_variants)

    grouped: dict[str, list[Match]] = {}
    for match in matches:
        grouped.setdefault(trace_variant(match), []).append(match)
    failed_grouped: dict[str, list[_FailedMatch]] = {}
    for failed in failed_matches:
        failed_grouped.setdefault(trace_variant(failed), []).append(failed)

    if variant_colors is None:
        variant_colors = variant_color_map(
            sorted(set(grouped) | set(failed_grouped), key=variant_sort_key)
        )

    # Column (estimator, x_variant) -> slowest (smallest) y among real matches,
    # so failed runs can be plotted just below the column they belong to.
    column_min_y: dict[tuple[str, str], float] = {}
    for match in matches:
        column = (_estimator_name(match), x_variant(match))
        y = math.log2(match.speedup)
        column_min_y[column] = min(column_min_y.get(column, y), y)

    all_values = [math.log2(match.speedup) for match in matches]
    fallback_min_y = min(all_values) if all_values else 0.0

    fig = go.Figure()
    for variant in sorted(set(grouped) | set(failed_grouped), key=variant_sort_key):
        color = variant_colors.get(variant)
        # A single dummy point per variant drives the legend swatch, so it always
        # shows the variant's real color/symbol regardless of the per-point
        # grey-out (sklearnex fallback) or "x" (failed run) styling below.
        fig.add_trace(
            go.Scatter(
                mode="markers",
                name=variant,
                legendgroup=variant,
                showlegend=True,
                hoverinfo="skip",
                x=[None],
                y=[None],
                marker={"size": 10, "symbol": "circle", "color": color},
            )
        )

        variant_matches = sorted(grouped.get(variant, []), key=_trace_sort_key)
        if variant_matches:
            marker = {
                "size": 10,
                "symbol": [_marker_symbol(match) for match in variant_matches],
                "color": [
                    _marker_color(match, color) for match in variant_matches
                ],
            }
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    name=variant,
                    legendgroup=variant,
                    showlegend=False,
                    x=[
                        estimator_positions[_estimator_name(match)]
                        + offsets[x_variant(match)]
                        for match in variant_matches
                    ],
                    y=[math.log2(match.speedup) for match in variant_matches],
                    text=[_hover_text(match) for match in variant_matches],
                    hovertemplate="%{text}<extra></extra>",
                    marker=marker,
                )
            )

        variant_failed = failed_grouped.get(variant, [])
        if variant_failed:
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    name=variant,
                    legendgroup=variant,
                    showlegend=False,
                    x=[
                        estimator_positions[_estimator_name(failed)]
                        + offsets[x_variant(failed)]
                        for failed in variant_failed
                    ],
                    y=[
                        column_min_y.get(
                            (_estimator_name(failed), x_variant(failed)),
                            fallback_min_y,
                        )
                        - FAILED_MARKER_GAP
                        for failed in variant_failed
                    ],
                    text=[
                        _failed_hover_text(failed.matched_result)
                        for failed in variant_failed
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    marker={"size": 10, "symbol": "x", "color": color},
                )
            )

    values = all_values + [
        column_min_y.get((_estimator_name(failed), x_variant(failed)), fallback_min_y)
        - FAILED_MARKER_GAP
        for failed in failed_matches
    ]
    min_tick = math.floor(min(min(values, default=0.0), 0))
    max_tick = math.ceil(max(max(values, default=0.0), 0))
    tick_values = list(range(min_tick, max_tick + 1))
    marker_notes = _marker_notes_html(matches, failed_count=len(failed_matches))
    fig.update_layout(
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(len(estimators))),
            "ticktext": [_abbreviate_estimator_label(e) for e in estimators],
            "range": [-0.5, len(estimators) - 0.5],
        },
        yaxis={
            "title": f"speed-up vs {baseline_label}",
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": [_format_speedup_tick(2**tick) for tick in tick_values],
        },
        shapes=[
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
        margin={"l": 70, "r": 20, "t": 20, "b": 110},
        showlegend=True,
        legend={"orientation": "h"},
        template="none",
    )
    fragment = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="540px",
        div_id=chart_id,
    )
    return f"{fragment}\n{marker_notes}"
