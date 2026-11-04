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
    elements: list[str],
    *,
    variant_colors: dict[str, str] | None = None,
    labels: list[str] | None = None,
    baseline_label: str | None = None,
):
    if not elements:
        return ""
    if variant_colors is None:
        variant_colors = {}
    tabs_id = f"tabs-{next(chart_ids)}"
    buttons = []
    panels = []
    for index, element in enumerate(elements):
        if labels is not None:
            label = labels[index]
        else:
            match = re.search(r"<h3>(.*?)</h3>", element)
            label = match.group(1) if match else f"Environment {index + 1}"
        marker = f"tab-{tabs_id}-{index}"
        style = ""
        color = variant_colors.get(label)
        if color is not None:
            style = f"background: {color}1F;"
        buttons.append(
            {
                "active": index == 0,
                "label": label,
                "marker": marker,
                "style": style,
                "is_baseline": label == baseline_label,
            }
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


def render_hardware_tabs(pages: list[tuple[str, str | None]]) -> str:
    """Pages whose html is None have nothing to show and get no tab."""
    pages = [(label, html) for label, html in pages if html is not None]
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


