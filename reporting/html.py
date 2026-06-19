from html import escape
import itertools
import json
import math
import re
from statistics import median

from jinja2 import Template

from .matching import Match

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

BASE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>sklbench dashboard</title>
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
      grid-template-columns: minmax(180px, 260px) 1fr;
      gap: 12px;
    }
    .package-list {
      columns: 2;
      margin: 0;
      padding-left: 18px;
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
  <h1>sklbench speed-up dashboard</h1>
  {% for row in rows %}
  <div class="page-row">{{ row }}</div>
  {% endfor %}
</body>
</html>
""")
BASE_TEMPLATE.globals["plotly_cdn"] = PLOTLY_CDN

DATE_RANGE_TEMPLATE = Template("""<section class="panel">
  <h2>Benchmark Window</h2>
  <p><strong>{{ label }}</strong></p>
  <p class="muted">{{ count }} result records{% if start and end %}, from {{ start }} to {{ end }}{% endif %}</p>
</section>""")

HARDWARE_TEMPLATE = Template("""<section class="panel">
  <h2>Hardware</h2>
  <div class="summary-grid">
    <div>
      <h3>CPU</h3>
      <p>{{ cpu_name }}</p>
      <p class="muted">{{ architecture }}, {{ logical_cpus }} logical CPUs</p>
      <p class="muted">{{ ram_gb }} GB RAM</p>
    </div>
    <div>
      <h3>GPU(s)</h3>
      {% if gpus %}
      <ul class="compact">
      {% for gpu in gpus %}
        <li>{{ gpu.name }} <span class="muted">({{ gpu.vendor }}, {{ gpu.memory_gb }} GB, driver {{ gpu.driver }})</span></li>
      {% endfor %}
      </ul>
      {% else %}
      <p class="muted">No GPU detected.</p>
      {% endif %}
    </div>
  </div>
</section>""")

SOFTWARE_TEMPLATE = Template("""<section class="software-details">
  <div>
    <h3>{{ name }}</h3>
    <p>{{ summary.implementation_label }}</p>
    <p class="muted">pixi environment: <code>{{ summary.environment }}</code></p>
  </div>
  <div>
    <h3>Packages</h3>
    <ul class="package-list">
    {% for package in summary.packages %}
      <li><code>{{ package.name }}</code> {{ package.version }}</li>
    {% endfor %}
    </ul>
    {% if summary.threadpools %}
    <h3>Threadpools</h3>
    <ul class="compact">
    {% for threadpool in summary.threadpools %}
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


def _hover_text(match: Match) -> str:
    result = match.matched_result
    base = match.base_result
    algorithm = result.case.get("algorithm", {})
    data = result.case.get("data", {})
    warning_lines = [f"{warning.icon} {warning.message}" for warning in match.warnings]
    lines = [
        f"<b>{escape(algorithm.get('estimator', 'unknown'))}</b>",
        f"target: {escape(result.implementation.short_name)}",
        f"base median: {median(base.times):.3g} ms",
        f"target median: {median(result.times):.3g} ms",
        f"speed-up: {match.speedup:.3g}x",
    ]
    try:
        metrics_match = match.metrics_match
    except ValueError:
        metrics_match = False
    if not metrics_match:
        lines.append("<b>metrics differ</b>")
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


def _metrics_match(match: Match) -> bool:
    try:
        return match.metrics_match
    except ValueError:
        return False


def _point_has_max_bins(match: Match) -> bool:
    estimator_params = match.matched_result.case.get("algorithm", {}).get(
        "estimator_params", {}
    )
    return isinstance(estimator_params, dict) and "max_bins" in estimator_params


def _mixed_max_bins_columns(matches: list[Match]) -> set[tuple[str, str]]:
    column_has_max_bins = {}
    for match in matches:
        estimator = match.matched_result.case.get("algorithm", {}).get(
            "estimator", "unknown"
        )
        variant = match.matched_result.implementation.short_name
        column_has_max_bins.setdefault((estimator, variant), set()).add(
            _point_has_max_bins(match)
        )
    return {
        key
        for key, has_max_bins_values in column_has_max_bins.items()
        if has_max_bins_values == {False, True}
    }


def plotly_colored_tabs(elements: list[str]):
    if not elements:
        return ""
    tabs_id = f"tabs-{next(_chart_ids)}"
    buttons = []
    panels = []
    for index, element in enumerate(elements):
        active = " active" if index == 0 else ""
        match = re.search(r"<h3>(.*?)</h3>", element)
        label = match.group(1) if match else f"Environment {index + 1}"
        marker = f"tab-{tabs_id}-{index}"
        buttons.append(
            f'<button class="tab-button{active}" type="button" '
            f'data-tab-target="{marker}">{escape(label)}</button>'
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


def assemble_plots_in_grid(plots: list[dict], *, x=None, y=None, rows=None, columns=None):
    row_key = rows or y
    column_key = columns or x
    if row_key is None or column_key is None:
        raise TypeError("assemble_plots_in_grid requires rows/columns or x/y")
    if not plots:
        return '<section class="empty">No matching benchmark results.</section>'

    row_values = sorted({plot[row_key] for plot in plots})
    column_values = sorted({plot[column_key] for plot in plots})
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


def speedup_plot_html(matches: list[Match]):
    if not matches:
        return '<div class="empty">No matches for this group.</div>'

    chart_id = f"chart-{next(_chart_ids)}"
    estimators = sorted(
        {
            match.matched_result.case.get("algorithm", {}).get("estimator", "unknown")
            for match in matches
        }
    )
    estimator_positions = {
        estimator: index for index, estimator in enumerate(estimators)
    }
    variants = sorted({match.matched_result.implementation.short_name for match in matches})
    offsets = _variant_offsets(variants)
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    variant_colors = {
        variant: palette[index % len(palette)] for index, variant in enumerate(variants)
    }
    grouped: dict[str, list[Match]] = {}
    for match in matches:
        grouped.setdefault(match.matched_result.implementation.short_name, []).append(match)
    mixed_max_bins_columns = _mixed_max_bins_columns(matches)

    traces = []
    for variant, variant_matches in sorted(grouped.items()):
        variant_matches = sorted(
            variant_matches,
            key=lambda match: match.matched_result.case.get("algorithm", {}).get(
                "estimator", "unknown"
            ),
        )
        marker_symbols = []
        for match in variant_matches:
            estimator = match.matched_result.case.get("algorithm", {}).get(
                "estimator", "unknown"
            )
            max_bins_is_comparison_axis = (
                estimator,
                match.matched_result.implementation.short_name,
            ) in mixed_max_bins_columns
            metrics_match = _metrics_match(match)
            marker_symbols.append(
                "x"
                if _point_has_max_bins(match) and max_bins_is_comparison_axis
                else "diamond-open"
                if not metrics_match
                else "circle"
            )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": variant,
                "x": [
                    estimator_positions[
                        match.matched_result.case.get("algorithm", {}).get(
                            "estimator", "unknown"
                        )
                    ]
                    + offsets[variant]
                    for match in variant_matches
                ],
                "y": [math.log2(match.speedup) for match in variant_matches],
                "text": [_hover_text(match) for match in variant_matches],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "size": 10,
                    "symbol": marker_symbols,
                    "color": variant_colors[variant],
                },
            }
        )

    values = [math.log2(match.speedup) for match in matches]
    min_tick = math.floor(min(min(values), 0))
    max_tick = math.ceil(max(max(values), 0))
    tick_values = list(range(min_tick, max_tick + 1))
    layout = {
        "xaxis": {
            "title": "Estimator",
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
        "legend": {"orientation": "h"},
    }
    return f"""<div id="{chart_id}" class="chart"></div>
<script>
  Plotly.newPlot("{chart_id}", {_safe_json(traces)}, {_safe_json(layout)}, {{responsive: true}});
</script>"""
