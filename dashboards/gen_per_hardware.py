from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.utils import partition_iterable, groupby

from sklbench.reporting.matching import (
    append_iterations_warning, append_max_bins_warning, read_all_results,
    find_matches, date_range, Match, MatchWarning, MethodResult,
    append_cpu_fallback_warning,
)

from sklbench.reporting.envs import read_env, summarize_software_env, summarize_hardware_env
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    assemble_plots_in_grid,
    detailed_results_table_html,
    speedup_plot_html,
    render_software_tabs,
    render_hardware_tabs,
    variant_color_map,
)


HARDWARE_NAMES = {
    "0f5327": "AMD 48 CPU cores",
    "4cb66b": "AMD CPU with Nvidia L4 GPU",
    "268063": "Intel laptop with B390 GPU",
}
BASE_IMPLEMENTATION = "sklearn"


def result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list[MatchWarning]]:
    """
    Assumptions:
    - hardware matches
    - candidate implementation is not sklearn
    - base_res implementation is sklearn

    returns:
    - True/False
    - warnings
    """
    assert base_res.hardware_hash == candidate.hardware_hash
    assert base_res.implementation.short_name == BASE_IMPLEMENTATION
    assert candidate.implementation.short_name != BASE_IMPLEMENTATION

    warnings = []

    if candidate.is_sklearnex_tree:
        append_max_bins_warning(base_res, candidate, warnings)
    append_iterations_warning(base_res, candidate, warnings)
    append_cpu_fallback_warning(candidate, warnings)

    # TODO? warning for attributes:
    # - tree structure

    return (
        base_res.minimal_match_key == candidate.minimal_match_key,
        warnings
    )


def render_hardware_page(results: list[MethodResult], hardware_hash: str) -> str:
    results = [res for res in results if res.hardware_hash == hardware_hash]
    if not results:
        return '<section class="empty">No benchmark results for this hardware.</section>'
    hardwares_set = {res.hardware_hash for res in results}
    if len(hardwares_set) > 1:
        raise ValueError(f"Results are dirty: several hardware hashes match {hardware_hash!r}")

    base_results, other_results = partition_iterable(
        results,
        predicate=lambda res: res.implementation.short_name == BASE_IMPLEMENTATION
    )
    if not base_results:
        return f'<section class="empty">No {BASE_IMPLEMENTATION} baseline results for this hardware.</section>'

    variant_colors = variant_color_map(
        sorted({res.implementation.short_name for res in other_results})
    )
    grouped_results = groupby(base_results, lambda res: (res.category, res.method))

    plots = []
    matches_by_category = {}
    for (category, method), group_base_results in grouped_results.items():
        matches = find_matches(group_base_results, other_results, result_matches)
        matches_by_category.setdefault(category, {})[method] = matches
        # create a JS snippet for plotly:
        plots.append({
            "category": category,
            "method": method,
            "point_count": len(matches),
            "plot": speedup_plot_html(matches, variant_colors=variant_colors)
        })
    details_by_category = {
        category: detailed_results_table_html(
            category,
            method_matches,
            baseline_label=BASE_IMPLEMENTATION,
            variant_label=lambda result: result.implementation.short_name,
        )
        for category, method_matches in matches_by_category.items()
    }

    hardware_hash, = hardwares_set
    hardware_env = read_env("hardware", hardware_hash)

    base_sw_env = read_env("software", base_results[0].software_hash)
    base_implem = base_results[0].implementation

    softwares = [
        summarize_software_env(
            base_sw_env,
            base_implem,
            software_hash=base_results[0].software_hash,
        )
    ]
    for implem_name, implem_results in groupby(other_results, lambda res: res.implementation.short_name).items():
        res = implem_results[0]
        env = read_env("software", res.software_hash)
        softwares.append(
            summarize_software_env(
                env,
                res.implementation,
                software_hash=res.software_hash,
            )
        )

    rows = [
        DATE_RANGE_TEMPLATE.render(date_range(results)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(hardware_env)),
        render_software_tabs([
            SOFTWARE_TEMPLATE.render(**summary)
            for summary in softwares
        ], variant_colors=variant_colors),
        assemble_plots_in_grid(
            plots,
            rows={"category": ["linear", "tree-based", "clustering"]},
            columns={"method": ["fit", "predict"]},
            details_by_row=details_by_category,
        )
    ]
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


if __name__ == "__main__":
    results = read_all_results()
    hardware_pages = [
        (hardware_name, render_hardware_page(results, hardware_hash))
        for hardware_hash, hardware_name in HARDWARE_NAMES.items()
    ]

    html = BASE_TEMPLATE.render(rows=[
        render_hardware_tabs(hardware_pages),
    ])

    output = dashboard_output_path("per_hardware.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
