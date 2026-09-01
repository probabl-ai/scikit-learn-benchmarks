from html import escape
import itertools
import re

from ..envs import read_env, summarize_hardware_env, summarize_software_env
from ..matching import MethodResult
from .templates import (
    HARDWARE_TABS_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TABS_TEMPLATE,
    SOFTWARE_TEMPLATE,
)


chart_ids = itertools.count()


def _slug_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "tab"


def render_software_tabs(
    elements: list[str], *, variant_colors: dict[str, str] | None = None
):
    if not elements:
        return ""
    if variant_colors is None:
        variant_colors = {}
    tabs_id = f"tabs-{next(chart_ids)}"
    buttons = []
    panels = []
    for index, element in enumerate(elements):
        match = re.search(r"<h3>(.*?)</h3>", element)
        label = match.group(1) if match else f"Environment {index + 1}"
        marker = f"tab-{tabs_id}-{index}"
        style = ""
        color = variant_colors.get(label)
        if color is not None:
            style = f"background: {color}1F;"
        buttons.append(
            {"active": index == 0, "label": label, "marker": marker, "style": style}
        )
        panels.append({"active": index == 0, "html": element, "marker": marker})
    return SOFTWARE_TABS_TEMPLATE.render(
        buttons=buttons,
        panels=panels,
        tabs_id=tabs_id,
    )


def render_software_hardware_tabs(
    baseline_results: list[MethodResult],
    comparison_results: list[MethodResult],
    *,
    baseline_label: str,
    comparison_label,
    comparison_sort_key=None,
    variant_colors: dict[str, str] | None = None,
) -> str:
    elements = []
    if baseline_results:
        baseline = baseline_results[0]
        software_summary = summarize_software_env(
            read_env("software", baseline.software_hash),
            baseline.implementation,
            software_hash=baseline.software_hash,
        )
        software_summary["name"] = baseline_label
        hardware_summary = summarize_hardware_env(
            read_env("hardware", baseline.hardware_hash)
        )
        elements.append(
            SOFTWARE_TEMPLATE.render(**software_summary)
            + HARDWARE_TEMPLATE.render(hardware_summary)
        )

    def sort_key(result: MethodResult):
        label = comparison_label(result)
        if comparison_sort_key is None:
            return label
        return comparison_sort_key(label)

    seen_labels = set()
    for result in sorted(comparison_results, key=sort_key):
        label = comparison_label(result)
        if label in seen_labels:
            continue
        seen_labels.add(label)

        software_summary = summarize_software_env(
            read_env("software", result.software_hash),
            result.implementation,
            software_hash=result.software_hash,
        )
        software_summary["name"] = label
        hardware_summary = summarize_hardware_env(
            read_env("hardware", result.hardware_hash)
        )
        elements.append(
            SOFTWARE_TEMPLATE.render(**software_summary)
            + HARDWARE_TEMPLATE.render(hardware_summary)
        )

    return render_software_tabs(elements, variant_colors=variant_colors)


def render_hardware_tabs(pages: list[tuple[str, str]]) -> str:
    if not pages:
        return '<section class="empty">No matching benchmark results.</section>'
    tabs_id = "hardware-tabs"
    buttons = []
    panels = []
    for index, (label, html) in enumerate(pages):
        marker = _slug_id(label)
        buttons.append({"active": index == 0, "label": label, "marker": marker})
        panels.append({"active": index == 0, "html": html, "marker": marker})
    return HARDWARE_TABS_TEMPLATE.render(
        buttons=buttons,
        panels=panels,
        tabs_id=tabs_id,
    )


def _plot_cell_html(title: str, plot_html: str, *, method: str | None = None) -> str:
    attrs = f' data-method="{method}"' if method in ("fit", "predict") else ""
    toggle = ""
    if method == "predict":
        toggle = (
            '<button type="button" class="predict-collapse-toggle" '
            'aria-expanded="true" aria-label="Collapse predict plots" '
            'title="Collapse predict plots">&rsaquo;</button>'
        )
    return (
        f'<section class="plot-cell"{attrs}>'
        '<div class="plot-cell-head">'
        f'<h3 class="plot-cell-title">{escape(title)}</h3>{toggle}'
        "</div>"
        f'<div class="plot-cell-body">{plot_html}</div>'
        "</section>"
    )


def assemble_plots_in_grid(
    plots: list[dict],
    *,
    rows=None,
    columns=None,
    details_by_row=None,
    details_after_grid=None,
):
    row_key = rows if isinstance(rows, str) else list(rows)[0]
    column_key = columns if isinstance(columns, str) else list(columns)[0]

    if not plots:
        return '<section class="empty">No matching benchmark results.</section>'

    row_values = (
        sorted({plot[row_key] for plot in plots})
        if isinstance(rows, str)
        else rows[row_key]
    )
    column_values = (
        sorted({plot[column_key] for plot in plots})
        if isinstance(columns, str)
        else columns[column_key]
    )
    by_position = {
        (plot[row_key], plot[column_key]): plot
        for plot in plots
    }
    if details_by_row is None:
        details_by_row = {}
    if details_after_grid is None:
        details_after_grid = []
    cells = []
    for row_value in row_values:
        for column_value in column_values:
            plot = by_position.get((row_value, column_value))
            if plot is None:
                # column/row value doubles as the method when the grid is split
                # by fit/predict, so an empty cell still collapses in step with
                # its populated neighbors.
                empty_method = column_value if column_key == "method" else (
                    row_value if row_key == "method" else None
                )
                empty_attrs = (
                    f' data-method="{empty_method}"'
                    if empty_method in ("fit", "predict")
                    else ""
                )
                cells.append(
                    f'<section class="plot-cell empty"{empty_attrs}>No data</section>'
                )
                continue
            title = f"{row_value} / {column_value}"
            point_count = plot.get("point_count")
            if point_count is not None:
                point_label = "point" if point_count == 1 else "points"
                title = f"{title} ({point_count} {point_label})"
            cells.append(
                _plot_cell_html(title, plot["plot"], method=plot.get("method"))
            )
        detail = details_by_row.get(row_value)
        if detail:
            cells.append(f'<section class="plot-detail-row">{detail}</section>')

    grid_class = "plot-grid"
    extra_style = ""
    if column_key == "method" and "predict" in column_values:
        grid_class += " method-grid"
        collapsed_template = " ".join(
            "44px" if value == "predict" else "1fr" for value in column_values
        )
        extra_style = f" --collapsed-columns: {collapsed_template};"

    style = (
        f"grid-template-columns: repeat({len(column_values)}, minmax(0, 1fr));"
        f"{extra_style}"
    )
    grid = f'<section class="{grid_class}" style="{style}">{"".join(cells)}</section>'
    return grid + "".join(details_after_grid)
