from collections import defaultdict
from html import escape
import itertools
import json
import math

from plotly import graph_objects as go

from ..matching import BenchmarkRecord
from .table import data_dtype_label, data_layout_label


# scikit-learn's cyan and orange (doc/scss/colors.scss) lead, followed by
# hues picked to stay distinguishable from them and from the grey of fallback points.
SERIES_COLORS = [
    "#2294c4",
    "#f7931e",
    "#8045e5",
    "#2e9e5b",
    "#d1495b",
    "#15688c",
    "#b76c13",
    "#e377c2",
    "#9aa500",
    "#7ac5ec",
]

TEXT_COLOR = "#222832"
GRID_COLOR = "#e5e7ea"
REFERENCE_LINE_COLOR = "#48566b"
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

PLOT_TEMPLATE = go.layout.Template(
    layout={
        "font": {"family": FONT_FAMILY, "color": TEXT_COLOR, "size": 12},
        "colorway": SERIES_COLORS,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "hoverlabel": {"font": {"family": FONT_FAMILY}},
        "xaxis": {"gridcolor": GRID_COLOR, "linecolor": "#d1d5da", "zerolinecolor": GRID_COLOR},
        "yaxis": {"gridcolor": GRID_COLOR, "linecolor": "#d1d5da", "zerolinecolor": GRID_COLOR},
    }
)


chart_ids = itertools.count()


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


# Shortens tree-estimator names on x-axis tick labels only - everywhere else
# (hover text, warning tooltips, matching) keeps the full estimator name.
ESTIMATOR_LABEL_ABBREVIATIONS = {
    "HistGradientBoosting": "HGB",
    "RandomForest": "RF",
    "Regressor": "Reg",
    "Classifier": "Clf",
}


def _data_desc_hover_lines(case: dict, data_desc: dict) -> list[str]:
    """The data the timed method was actually handed, as recorded at run
    time."""
    lines = []
    samples, features = data_desc.get("samples"), data_desc.get("features")
    if samples is not None:
        shape = f"{samples:,}" if features is None else f"{samples:,} x {features}"
        lines.append(f"shape: {shape}")
    layout = data_layout_label(case, data_desc)
    if layout:
        lines.append(f"memory layout: {layout}")
    dtype = data_dtype_label(case, data_desc)
    if dtype:
        lines.append(f"dtype: {dtype}")
    return lines


def _record_fit_data_desc(record: BenchmarkRecord) -> dict | None:
    for run in record.runs:
        fit_desc = (run.get("data_desc") or {}).get("fit")
        if fit_desc:
            return fit_desc
    return None


def _data_hover_lines(case: dict, data_desc: dict | None) -> list[str]:
    data = case.get("data", {})
    if not data_desc:
        return _hover_lines(data)
    # The requested order/dtype, superseded by the recorded ones below.
    requested = {key: value for key, value in data.items() if key not in ("order", "dtype")}
    return _hover_lines(requested) + _data_desc_hover_lines(case, data_desc)


def case_hover_html(case: dict, data_desc: dict | None) -> str:
    """The part of a point's hover text that only depends on its own case."""
    algorithm = case.get("algorithm", {})
    estimator_params = "<br>".join(
        escape(line) for line in _hover_lines(algorithm.get("estimator_params", {}))
    )
    data_params = "<br>".join(
        escape(line) for line in _data_hover_lines(case, data_desc)
    )
    return "<br>".join(
        ["<br><b>estimator params</b>", estimator_params, "<br><b>data params</b>", data_params]
    )


def failed_hover_html(record: BenchmarkRecord) -> str:
    return "<br>".join(
        [
            "<b>benchmark run failed</b>",
            case_hover_html(record.case, _record_fit_data_desc(record)),
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
        variant: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, variant in enumerate(variants)
    }


def phase_breakdown_plot_html(
    points: list[dict],
    *,
    phase_order: list[str],
    phase_colors: dict[str, str],
    phase_labels: dict[str, str] | None = None,
    x_title: str = "threads",
    series_order: list[str] | None = None,
) -> str:
    """Stacked bar of phase timings vs. an x-axis category (e.g. thread
    count), one bar per `points` entry, plotted in seconds. Each point is
    `{"x": ..., "phases": {phase_name: ms}, "total_ms": ...}`, with an
    optional `"x_label"` overriding `str(x)` as the tick label (e.g. to show
    an actual-vs-requested thread count as `"4 (3)"`) while `x` itself still
    drives sort order. Legend is disabled on every trace - callers render one
    shared legend across a grid of these (see dashboards/gen_hgb_scalability_breakdown.py)
    rather than repeating it per small multiple.

    A point may also carry a `"series"` label (e.g. a build being compared
    against another) - when more than one distinct series is present, each
    `x` position gets one side-by-side bar per series (`offsetgroup` per
    series, stacked manually
    via each trace's `base` since `barmode="overlay"` doesn't stack on its
    own), ordered left-to-right by `series_order` (defaults to alphabetical).
    X-axis ticks stay plain `x`/`x_label` values either way - the series
    only affects a bar's left/right position at a given tick, not the tick
    itself. A single series (the common case, `"series"` absent everywhere)
    renders exactly as before - one plain stacked bar per `x`."""
    if phase_labels is None:
        phase_labels = {phase: phase for phase in phase_order}
    chart_id = f"phase-breakdown-{next(chart_ids)}"
    points = sorted(points, key=lambda point: point["x"])
    series_values = {point.get("series") for point in points if point.get("series") is not None}
    multi_series = len(series_values) > 1
    if multi_series and series_order is None:
        series_order = sorted(series_values)

    fig = go.Figure()
    for series in series_order if multi_series else [None]:
        series_points = (
            [point for point in points if point.get("series") == series]
            if multi_series
            else points
        )
        x_values = [str(point.get("x_label", point["x"])) for point in series_points]
        cumulative_ms = [0.0] * len(series_points)
        for phase in phase_order:
            y_values = [point["phases"].get(phase, 0.0) for point in series_points]
            totals = [point["total_ms"] for point in series_points]
            fig.add_trace(
                go.Bar(
                    name=phase_labels[phase],
                    x=x_values,
                    y=[y / 1000 for y in y_values],
                    base=[base / 1000 for base in cumulative_ms] if multi_series else None,
                    offsetgroup=series if multi_series else None,
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
                        (f"{series}<br>" if multi_series else "")
                        + f"{phase_labels[phase]}: "
                        "%{customdata[0]} (%{customdata[1]:.0%})<extra></extra>"
                    ),
                )
            )
            cumulative_ms = [total + y for total, y in zip(cumulative_ms, y_values)]
    fig.update_layout(
        barmode="overlay" if multi_series else "stack",
        xaxis={"type": "category", "title": x_title},
        yaxis={"title": "time (s)", "rangemode": "tozero"},
        margin={"l": 60, "r": 15, "t": 15, "b": 44},
        showlegend=False,
        template=PLOT_TEMPLATE,
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="320px",
        div_id=chart_id,
    )


# Rendered marker *diameter* (pixels) an optional per-point size metric is
# scaled into - see `scaling_line_plot_html`'s `size_domain`. Max trimmed to
# about 80% of its prior radius (17.6 vs. what a 22px-diameter/11px-radius
# max used to be) so a 100%-CPU point doesn't dominate a small-multiple cell.
_MARKER_DIAMETER_RANGE_PX = (6, 17.6)


def _marker_sizes(points: list[tuple], size_domain: tuple[float, float]) -> list[float] | None:
    """Per-point marker *areas* (pixels^2, for `marker.sizemode="area"`) from
    each point's optional 4th element, linearly mapped from `size_domain` -
    so e.g. 50% CPU utilization reads as half the marker *area* (matching how
    area is actually perceived), not half the diameter, which `sizemode`'s
    default would otherwise give and which visually compresses low values
    against high ones. `_MARKER_DIAMETER_RANGE_PX` bounds are converted to
    the equivalent area range here, then linearly interpolated (clamped, so
    an out-of-domain value doesn't shrink/grow past the range) - or `None`
    if no point in this series carries a value, leaving Plotly's own default
    marker size untouched."""
    if not any(len(point) > 3 and point[3] is not None for point in points):
        return None
    lo, hi = size_domain
    min_d, max_d = _MARKER_DIAMETER_RANGE_PX
    min_area, max_area = math.pi * (min_d / 2) ** 2, math.pi * (max_d / 2) ** 2
    sizes = []
    for point in points:
        value = point[3] if len(point) > 3 else None
        if value is None:
            sizes.append(min_area)
            continue
        fraction = 0.5 if hi <= lo else (value - lo) / (hi - lo)
        fraction = min(1.0, max(0.0, fraction))
        sizes.append(min_area + fraction * (max_area - min_area))
    return sizes


def scaling_line_plot_html(
    series: dict[str, list[tuple[float, float]]],
    *,
    colors: dict[str, str] | None = None,
    line_dashes: dict[str, str] | None = None,
    x_title: str = "threads",
    y_title: str = "fit time (ms)",
    y_unit: str = "ms",
    x_log: bool = False,
    y_log: bool = False,
    reference_lines: dict[str, list[list[tuple[float, float]]]] | None = None,
    size_domain: tuple[float, float] = (0, 100),
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
    context an aggregate x/y trend can't convey. A point may further add a
    fourth element, `""`/`None` or a metric value (e.g. CPU utilization %)
    to size that point's marker *area* by - `size_domain` (same units, fixed
    rather than derived per-series so e.g. 50% CPU utilization renders as
    the same marker size in every cell of a grid, not rescaled per-cell)
    maps it into `_MARKER_DIAMETER_RANGE_PX` (see `_marker_sizes`); points
    with no 4th element anywhere in a series keep Plotly's own default
    marker size.

    `reference_lines` draws additional dashed, marker-less, grey lines (e.g.
    an ideal-scaling reference), kept visually distinct from the real data
    traces. Each label maps to a list of segments (`[[(x, y), ...], ...]`),
    drawn as one trace with gaps between them, so several same-meaning
    segments share one legend entry.

    `line_dashes` maps a series label to a Plotly dash style (e.g. "dash"),
    for series sharing a color that still need telling apart."""
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
        marker_sizes = _marker_sizes(points, size_domain)
        marker = {"color": color} if color else {}
        line = dict(marker)
        dash = (line_dashes or {}).get(label)
        if dash:
            line["dash"] = dash
        if marker_sizes is not None:
            # `_marker_sizes` returns areas (px^2) already, so `sizemode`
            # must be "area" too - Plotly's default `"diameter"` would
            # otherwise reinterpret them as diameters and square the effect.
            marker["size"] = marker_sizes
            marker["sizemode"] = "area"
            marker["sizeref"] = 1
        fig.add_trace(
            go.Scatter(
                name=label,
                x=x_values,
                y=y_values,
                mode="lines+markers",
                line=line,
                marker=marker,
                customdata=hover_extra,
                hovertemplate=(
                    f"{label}<br>%{{x}} {x_title}: %{{y:.3g}}{y_unit}"
                    "%{customdata}<extra></extra>"
                ),
                # A single series has nothing to contrast its label against
                # (e.g. hardware with no SMT cores never gets a "without
                # SMT" counterpart - see gen_models_scalability.py), so its
                # legend entry would just be noise.
                showlegend=len(series) > 1,
            )
        )
    for label, segments in sorted((reference_lines or {}).items()):
        x_values, y_values = [], []
        for segment in segments:
            segment = sorted(segment, key=lambda point: point[0])
            if x_values:
                x_values.append(None)
                y_values.append(None)
            x_values.extend(point[0] for point in segment)
            y_values.extend(point[1] for point in segment)
            all_x_values.update(point[0] for point in segment)
        fig.add_trace(
            go.Scatter(
                name=label,
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": REFERENCE_LINE_COLOR, "dash": "dash", "width": 1},
                hoverinfo="skip",
            )
        )
    xaxis = {"title": x_title}
    if x_log and all_x_values:
        min_x, max_x = min(all_x_values), max(all_x_values)
        start_exp = math.ceil(math.log2(max(min_x, 1)))
        end_exp = math.ceil(math.log2(max(max_x, 1)))
        tick_values = [2**exp for exp in range(start_exp, end_exp + 1)]
        # Tick a non-power-of-two first point (e.g. an n_jobs sweep starting
        # at 11) so the axis visibly doesn't start at 1.
        if min_x not in tick_values:
            tick_values.insert(0, min_x)
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
        template=PLOT_TEMPLATE,
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
    distinguishing them.

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
            "line": {"color": REFERENCE_LINE_COLOR, "width": 1, "dash": "dash"},
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
                "line": {"color": REFERENCE_LINE_COLOR, "width": 1, "dash": "dash"},
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
        template=PLOT_TEMPLATE,
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


