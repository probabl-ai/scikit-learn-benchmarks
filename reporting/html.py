from html import escape
import itertools
import json
import math
import re
from statistics import median

from jinja2 import Template

from .matching import Match

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
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

BASE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ title|default("sklbench dashboard") }}</title>
  <script src="{{ plotly_cdn }}"></script>
  <style>
    :root {
      color-scheme: light;
      --text: #222;
      --muted: #5f6368;
      --line: #d9dde3;
      --surface: #f7f8fa;
      --accent: #2864c8;
      --warn: #8a1f11;
    }
    body {
      margin: 24px;
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: white;
    }
    h1 { margin: 0 0 20px; font-size: 30px; }
    h2 { margin: 0 0 12px; font-size: 20px; }
    h3 { margin: 0 0 8px; font-size: 16px; }
    code {
      padding: 2px 4px;
      border-radius: 4px;
      background: #eef1f5;
    }
    .page-row {
      margin-bottom: 24px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: white;
    }
    .panel p {
      margin: 5px 0;
    }
    .muted {
      color: var(--muted);
    }
    .tabs {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: white;
    }
    .tab-buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 0;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }
    .tab-button {
      border: 0;
      border-right: 1px solid var(--line);
      padding: 10px 14px;
      background: transparent;
      color: var(--muted);
      font: inherit;
      cursor: pointer;
    }
    .tab-button.active {
      color: var(--text);
      background: white;
      box-shadow: inset 0 3px 0 var(--accent);
    }
    .tab-panel {
      display: none;
      padding: 16px;
    }
    .tab-panel.active {
      display: block;
    }
    .software-details {
      display: grid;
      grid-template-columns: max-content max-content max-content;
      gap: 12px 28px;
      align-items: start;
    }
    .package-list {
      columns: 2;
      column-gap: 40px;
      width: max-content;
      max-width: 100%;
      margin: 0 0 5px 0;
      padding-left: 18px;
    }
    @media (max-width: 900px) {
      .software-details {
        grid-template-columns: 1fr;
      }
      .package-list {
        width: auto;
      }
    }
    .plot-grid {
      display: grid;
      gap: 18px;
    }
    .plot-cell {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: white;
    }
    .plot-cell h3 {
      color: var(--muted);
      font-weight: 600;
    }
    .chart {
      width: 100%;
      height: 540px;
    }
    .empty {
      color: var(--muted);
      padding: 24px;
      text-align: center;
      background: var(--surface);
      border-radius: 8px;
    }
    ul.compact {
      margin: 6px 0 0;
      padding-left: 18px;
    }
  </style>
</head>
<body>
  <h1>{{ title|default("sklbench dashboard") }}</h1>
  {% for row in rows %}
  <div class="page-row">{{ row }}</div>
  {% endfor %}
</body>
</html>
""")
BASE_TEMPLATE.globals["plotly_cdn"] = PLOTLY_CDN

DATE_RANGE_TEMPLATE = Template("""<section class="panel">
  <h2>{{ label }}</h2>
</section>""")

HARDWARE_TEMPLATE = Template("""<section class="panel">
  <h2>Hardware</h2>
  <div class="summary-grid">
    <div>
      <h3>CPU</h3>
      <p>{{ cpu_name }}</p>
      <p class="muted">{{ architecture }}, {{ physical_cores }} physical cores, {{ logical_cpus }} logical CPUs</p>
      <p class="muted">{{ ram_gb }} GB RAM</p>
    </div>
    <div>
      <h3>GPU(s)</h3>
      {% if gpus %}
      <ul class="compact">
      {% for gpu in gpus %}
        <li><code>{{ gpu.id }}</code>: {{ gpu.name }} <span class="muted">({{ gpu.memory_gb }} GB)</span></li>
      {% endfor %}
      </ul>
      {% else %}
      <p class="muted">No GPU detected.</p>
      {% endif %}
    </div>
  </div>
</section>""")

SOFTWARE_TEMPLATE = Template("""<section class="software-details">
  <div style="min-width: 150px">
    <h3>{{ name }}</h3>
    <p> Python {{ python_version }}</p>
    {% if array_api_docs_url %}
    <p><a href="{{ array_api_docs_url }}">Array API support</a></p>
    {% endif %}
  </div>
  <div>
    <h3>Packages</h3>
    <ul class="package-list">
    {% for package in packages %}
      <li><code>{{ package.name }}</code> {{ package.version }} <span class="muted">({{ package.kind }})</span></li>
    {% endfor %}
    </ul>
  </div>
  <div>
    {% if threadpools %}
    <h3>Threadpools</h3>
    <ul class="compact">
    {% for threadpool in threadpools %}
      <li>{{ threadpool }}</li>
    {% endfor %}
    </ul>
    {% endif %}
  </div>
</section>""")

_chart_ids = itertools.count()


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


def _warning_tooltip_line(warning) -> str:
    message = warning.short_message or ""
    if not message:
        return warning.icon
    return f"{warning.icon} {message}"


def _warning_annotation_lines(matches: list[Match]) -> list[str]:
    warnings = []
    seen = set()
    for match in matches:
        for warning in match.warnings:
            key = (warning.icon, warning.message)
            if key in seen:
                continue
            seen.add(key)
            warnings.append(warning)
    return [
        f"{escape(warning.icon)}: {escape(warning.message)}"
        for warning in warnings
    ]

def custom_format(t):
    if t < 10:
        return f"{t:.2g}ms"
    elif t < 1000:
        return f"{round(t)}ms"
    elif t < 10_000:
        return f"{t/1000:.2g}s"
    else:
        return f"{round(t/1000)}s"


def _hover_text(match: Match) -> str:
    result = match.matched_result
    base = match.base_result
    algorithm = result.case.get("algorithm", {})
    data = result.case.get("data", {})
    warning_lines = [_warning_tooltip_line(warning) for warning in match.warnings]
    lines = [
        f"<b>speed-up: {match.speedup:.2g}x</b> "
        f"({custom_format(median(base.times))} vs {custom_format(median(result.times))})",
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
    estimator = match.matched_result.case.get("algorithm", {}).get(
        "estimator", "unknown"
    )
    return (_marker_symbol(match) != "circle", estimator)


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


def render_software_tabs(
    elements: list[str], *, variant_colors: dict[str, str] | None = None
):
    if not elements:
        return ""
    if variant_colors is None:
        variant_colors = {}
    tabs_id = f"tabs-{next(_chart_ids)}"
    buttons = []
    panels = []
    for index, element in enumerate(elements):
        active = " active" if index == 0 else ""
        match = re.search(r"<h3>(.*?)</h3>", element)
        label = match.group(1) if match else f"Environment {index + 1}"
        marker = f"tab-{tabs_id}-{index}"
        style = ""
        color = variant_colors.get(label)
        if color is not None:
            style = f' style="background: {color}1F;"'
        buttons.append(
            f'<button class="tab-button{active}" type="button" '
            f'data-tab-target="{marker}"{style}>{escape(label)}</button>'
        )
        panels.append(f'<div id="{marker}" class="tab-panel{active}">{element}</div>')
    return f"""<section class="tabs" id="{tabs_id}">
  <div class="tab-buttons">{''.join(buttons)}</div>
  {''.join(panels)}
  <script>
    document.querySelectorAll("#{tabs_id} .tab-button").forEach((button) => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll("#{tabs_id} .tab-button").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll("#{tabs_id} .tab-panel").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tabTarget).classList.add("active");
      }});
    }});
  </script>
</section>"""


def _slug_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "tab"


def render_hardware_tabs(pages: list[tuple[str, str]]) -> str:
    if not pages:
        return '<section class="empty">No matching benchmark results.</section>'
    tabs_id = "hardware-tabs"
    buttons = []
    panels = []
    for index, (label, html) in enumerate(pages):
        active = " active" if index == 0 else ""
        marker = _slug_id(label)
        buttons.append(
            f'<button class="tab-button{active}" type="button" '
            f'data-tab-target="{marker}">{escape(label)}</button>'
        )
        panels.append(f'<div id="{marker}" class="tab-panel{active}">{html}</div>')
    return f"""<section class="tabs" id="{tabs_id}">
  <div class="tab-buttons">{''.join(buttons)}</div>
  {''.join(panels)}
  <script>
    const hardwareTabButtons = document.querySelectorAll("#{tabs_id} > .tab-buttons .tab-button");
    function activateHardwareTab(button, updateHash = false) {{
      document.querySelectorAll("#{tabs_id} > .tab-buttons .tab-button").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll("#{tabs_id} > .tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const panel = document.getElementById(button.dataset.tabTarget);
      panel.classList.add("active");
      panel.querySelectorAll(".chart").forEach((chart) => Plotly.Plots.resize(chart));
      if (updateHash) {{
        history.replaceState(null, "", `#${{button.dataset.tabTarget}}`);
      }}
    }}
    hardwareTabButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        activateHardwareTab(button, true);
      }});
    }});
    const hardwareTabFromHash = Array.from(hardwareTabButtons).find(
      (button) => button.dataset.tabTarget === window.location.hash.slice(1)
    );
    if (hardwareTabFromHash) {{
      activateHardwareTab(hardwareTabFromHash);
    }}
  </script>
</section>"""


def assemble_plots_in_grid(plots: list[dict], *, rows=None, columns=None):
    row_key = rows if isinstance(rows, str) else list(rows)[0]
    column_key = columns if isinstance(columns, str) else list(columns)[0]

    if not plots:
        return '<section class="empty">No matching benchmark results.</section>'

    row_values = sorted({plot[row_key] for plot in plots}) if isinstance(rows, str) else rows[row_key]
    column_values = sorted({plot[column_key] for plot in plots}) if isinstance(columns, str) else columns[column_key]
    by_position = {
        (plot[row_key], plot[column_key]): plot
        for plot in plots
    }
    cells = []
    for row_value in row_values:
        for column_value in column_values:
            plot = by_position.get((row_value, column_value))
            if plot is None:
                cells.append('<section class="plot-cell empty">No data</section>')
                continue
            title = f"{row_value} / {column_value}"
            cells.append(
                '<section class="plot-cell">'
                f"<h3>{escape(title)}</h3>"
                f"{plot['plot']}"
                "</section>"
            )
    style = f"grid-template-columns: repeat({len(column_values)}, minmax(0, 1fr));"
    return f'<section class="plot-grid" style="{style}">{"".join(cells)}</section>'


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

    chart_id = f"chart-{next(_chart_ids)}"
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
    warning_annotation_lines = _warning_annotation_lines(matches)
    annotations = []
    bottom_margin = 110
    if warning_annotation_lines:
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
                "text": "<br>".join(warning_annotation_lines),
                "font": {"size": 12, "color": "#5f6368"},
            }
        )
        bottom_margin += 18 * len(warning_annotation_lines)
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
        "annotations": annotations,
        "margin": {"l": 70, "r": 20, "t": 20, "b": bottom_margin},
        "showlegend": True,
        "legend": {"orientation": "h"},
    }
    return f"""<div id="{chart_id}" class="chart"></div>
<script>
  Plotly.newPlot("{chart_id}", {_safe_json(traces)}, {_safe_json(layout)}, {{responsive: true}});
</script>"""
