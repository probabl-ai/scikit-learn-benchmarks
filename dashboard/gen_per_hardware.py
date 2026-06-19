from reporting.utils import partition_iterable, stable_json, groupby

from reporting.matching import (
    append_max_bins_warning, read_all_results, find_matches, date_range,
    Match, MatchWarning, Result
)

from reporting.configs import read_env, summarize_software_env, summarize_hardware_env
from reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    speedup_plot_html, plotly_colored_tabs, assemble_plots_in_grid
)


HARDWARE_NAME = "small-laptop"  # single hardware for now
BASE_IMPLEMENTATION = "sklearn"


def result_matches(base_res: Result, candidate: Result) -> tuple[bool, list[MatchWarning]]:
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

    case = base_res.case
    candidate_case = candidate.case
    warnings = []

    if candidate.is_sklearnex_tree:
        append_max_bins_warning(base_res, candidate, warnings)

    # TODO? warning for attributes:
    # - tree structure
    # - solver/n_iter

    return (
        stable_json(case) == stable_json(candidate_case),
        warnings
    )


if __name__ == "__main__":

    results = read_all_results()
    results = [res for res in results if res.hardware == HARDWARE_NAME]

    hardwares_set = {res.hardware_hash for res in results}
    if len(hardwares_set) > 1:
        raise ValueError("Results are dirty: several hardwares share the same name")

    base_results, other_results = partition_iterable(
        results,
        predicate=lambda res: res.implementation.short_name == BASE_IMPLEMENTATION
    )

    grouped_results = groupby(base_results, lambda res: (res.category, res.method))

    plots = []
    for (category, method), group_base_results in grouped_results.items():
        matches = find_matches(base_results, other_results, result_matches)
        # create a JS snippet for plotly:
        plots.append({
            "category": category,
            "method": method,
            "plot": speedup_plot_html(matches)
        })

    hardware_hash, = hardwares_set
    hardware_env = read_env("hardware", hardware_hash)

    base_sw_env = read_env("software", base_results[0].software_hash)
    base_implem = base_results[0].case['implementation']

    softwares = [
        (BASE_IMPLEMENTATION, summarize_software_env(base_sw_env, base_implem))
    ]
    for implem_name, implem_results in groupby(other_results, lambda res: res.implementation.short_name).items():
        res = implem_results[0]
        env = read_env("software", res.software_hash)
        softwares.append(
            (implem_name, summarize_software_env(env, res.implementation))
        )

    html = BASE_TEMPLATE.render(rows=[
        DATE_RANGE_TEMPLATE.render(date_range(results)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(hardware_env)),
        plotly_colored_tabs([
            SOFTWARE_TEMPLATE.render(sw)
            for sw in softwares
        ]),
        assemble_plots_in_grid(plots, rows="category", columns="method")
    ])

    # TODO: save html in dashboard/per_hardware.html
