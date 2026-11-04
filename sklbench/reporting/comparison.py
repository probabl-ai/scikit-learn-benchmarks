"""Speed-up grid and baseline switch shared by the software, hardware and cross
comparison dashboards. They differ only in what a variant is and in how results
of two variants are paired."""

from html import escape
from typing import Callable

from .html import assemble_plots_in_grid, detailed_results_table_html, speedup_plot_html
from .html.table import default_comparison_key
from .matching import (
    BenchmarkRecord,
    Match,
    MatchWarning,
    MethodResult,
    add_preprocessing_time,
    append_cpu_fallback_warning,
    append_cuml_rf_bins_warning,
    append_iterations_warning,
    append_max_bins_warning,
    append_solver_warning,
    find_matches,
    is_real_dataset,
)
from .utils import groupby, partition_iterable


PREPROCESSING_TOGGLE_HTML = (
    '<section class="panel">'
    '<label class="preprocessing-toggle">'
    '<input type="checkbox" class="preprocessing-toggle-checkbox">'
    " Include <dfn>preprocessing</dfn> time in fit speed-ups (real datasets only)"
    "</label>"
    "</section>"
)


def tree_binning_note(variant_noun: str) -> str:
    return (
        f"Tree-based plots: within each {variant_noun}, cases without <dfn>binning</dfn> "
        "(<dfn>exact splits</dfn>) are on the left, cases with binning "
        "(<dfn>histogram-based splits</dfn> or <dfn>quantile-based splits</dfn>) "
        "on the right."
    )


# The picked baseline is kept across outer tabs when that tab has it.
BASELINE_SWITCH_SCRIPT = """<script>
function sklbenchShowBaseline(baselineSwitch, baseline) {
  baselineSwitch.querySelectorAll(".baseline-view").forEach((view) => {
    view.classList.toggle("active", view.dataset.baseline === baseline);
  });
  baselineSwitch.querySelectorAll(".baseline-pick").forEach((button) => {
    const picked = button.dataset.baseline === baseline;
    button.disabled = picked;
    button.textContent = picked ? "✓ Current baseline" : "Pick as baseline";
    const panelId = button.closest(".tab-panel").id;
    baselineSwitch
      .querySelector(`.tab-button[data-tab-target="${panelId}"]`)
      .classList.toggle("is-baseline", picked);
  });
}
document.querySelectorAll(".baseline-switch").forEach((baselineSwitch) => {
  sklbenchShowBaseline(
    baselineSwitch, baselineSwitch.querySelector(".baseline-view.active").dataset.baseline
  );
});
document.addEventListener("click", (event) => {
  const pick = event.target.closest(".baseline-pick");
  if (!pick) {
    return;
  }
  const baseline = pick.dataset.baseline;
  document.querySelectorAll(".baseline-switch").forEach((baselineSwitch) => {
    const views = baselineSwitch.querySelectorAll(".baseline-view");
    if (Array.from(views).some((view) => view.dataset.baseline === baseline)) {
      sklbenchShowBaseline(baselineSwitch, baseline);
    }
  });
  document.querySelectorAll(".plotly-graph-div").forEach((chart) => {
    if (chart.offsetParent !== null) {
      Plotly.Plots.resize(chart);
    }
  });
  sklbenchWirePlotClicks();
});
</script>"""


def _max_bins(result: MethodResult) -> int | None:
    return result.case.get("algorithm", {}).get("estimator_params", {}).get("max_bins")


def result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list[MatchWarning]]:
    """`find_matches` match function. Either side can be sklearnex or run on a
    GPU, depending on the picked baseline."""
    warnings = []

    if base_res.is_sklearnex_tree and candidate.is_sklearnex_tree:
        # sklearnex tree cases are run both with the default max_bins and
        # with exact splits: pair each with its counterpart.
        if _max_bins(base_res) != _max_bins(candidate):
            return False, warnings
    # Against cuML, `append_cuml_rf_bins_warning` covers the binning
    # difference instead.
    elif candidate.is_sklearnex_tree:
        if base_res.implementation.library == "sklearn":
            append_max_bins_warning(base_res, candidate, warnings)
    elif base_res.is_sklearnex_tree and not base_res.is_sklearnex_fallback:
        if candidate.implementation.library == "sklearn":
            append_max_bins_warning(candidate, base_res, warnings)
    append_cuml_rf_bins_warning(base_res, candidate, warnings)
    append_iterations_warning(base_res, candidate, warnings)
    append_solver_warning(base_res, candidate, warnings)
    append_cpu_fallback_warning(base_res, warnings)
    append_cpu_fallback_warning(candidate, warnings)

    return True, warnings


def find_speedup_matches(
    base_results: list[MethodResult],
    other_results: list[MethodResult],
    match_key: Callable[[MethodResult], str] | None = None,
) -> list[Match]:
    # A sklearnex baseline has one tree result per max_bins value for the same
    # case. `find_matches` keeps a single baseline per case, so match each
    # max_bins value separately.
    base_by_max_bins: dict[int | None, list[MethodResult]] = {}
    for base in base_results:
        base_by_max_bins.setdefault(_max_bins(base), []).append(base)
    return [
        match
        for max_bins_base_results in base_by_max_bins.values()
        for match in find_matches(
            max_bins_base_results, other_results, result_matches, match_key=match_key
        )
    ]


def render_speedup_grid(
    results: list[MethodResult],
    failed_records: list[BenchmarkRecord],
    *,
    is_baseline: Callable[[MethodResult | BenchmarkRecord], bool],
    baseline_label: str,
    variant_label: Callable[[MethodResult | BenchmarkRecord], str],
    variant_colors: dict[str, str],
    empty_scope: str,
    match_key: Callable[[MethodResult], str] | None = None,
    comparison_key: Callable[[MethodResult | BenchmarkRecord], str] = default_comparison_key,
    plot_match: Callable[[Match], Match] = lambda match: match,
    plot_kwargs: dict | None = None,
    table_kwargs: dict | None = None,
    notes_by_row: dict[str, str] | None = None,
) -> str:
    """The category x fit/predict speed-up grid for one baseline, with a
    detailed results table per category. Non-baseline `results` are the plotted
    candidates. `match_key` defaults to `find_matches`'s, and `comparison_key`
    is the case identity linking plot points to table rows."""
    plot_kwargs = plot_kwargs or {}
    table_kwargs = table_kwargs or {}
    base_results, other_results = partition_iterable(results, predicate=is_baseline)
    if not base_results:
        return f'<section class="empty">No {baseline_label} baseline results for {empty_scope}.</section>'

    grouped_results = groupby(
        (res for res in base_results if res.method in ("fit", "predict")),
        lambda res: (res.category, res.method),
    )

    # Non-baseline (candidate) failures: shown on the plots too, at the bottom
    # of their model-variant column, since we don't know from a failed record
    # whether fit or predict is what failed.
    candidate_failed_records = [
        record for record in failed_records if not is_baseline(record)
    ]
    candidate_failed_by_category = groupby(
        candidate_failed_records, lambda record: record.category
    )

    plots = []
    matches_by_category = {}
    for (category, method), group_base_results in grouped_results.items():
        matches = find_speedup_matches(group_base_results, other_results, match_key)
        matches_by_category.setdefault(category, {})[method] = matches
        plot_failed_records = candidate_failed_by_category.get(category, [])
        plots.append({
            "category": category,
            "method": method,
            "case_count": len(
                {comparison_key(match.base_result) for match in matches}
                | {comparison_key(record) for record in plot_failed_records}
            ),
            "plot": speedup_plot_html(
                [plot_match(match) for match in matches],
                baseline_label=baseline_label,
                variant_colors=variant_colors,
                trace_variant=lambda match: variant_label(match.matched_result),
                comparison_key=comparison_key,
                failed_records=plot_failed_records,
                **plot_kwargs,
            )
        })
    failed_by_category = groupby(failed_records, lambda record: record.category)

    # A failed record means find_matches never sees a pair for that case, so the
    # side that did succeed (the base when a candidate failed, or any candidate
    # when the base itself failed) would silently vanish from the table too. Look
    # those up by case identity so they still show up, with no speed-up.
    base_by_case_key: dict[str, list[MethodResult]] = {}
    for base in base_results:
        base_by_case_key.setdefault(comparison_key(base), []).append(base)
    other_by_case_key: dict[str, list[MethodResult]] = {}
    for other in other_results:
        other_by_case_key.setdefault(comparison_key(other), []).append(other)

    unmatched_base_by_category: dict[str, list[MethodResult]] = {}
    unmatched_candidate_by_category: dict[str, list[MethodResult]] = {}
    for record in failed_records:
        key = comparison_key(record)
        if is_baseline(record):
            for candidate in other_by_case_key.get(key, []):
                unmatched_candidate_by_category.setdefault(candidate.category, []).append(candidate)
        else:
            for base in base_by_case_key.get(key, []):
                unmatched_base_by_category.setdefault(base.category, []).append(base)

    details_by_category = {
        category: detailed_results_table_html(
            category,
            matches_by_category.get(category, {}),
            baseline_label=baseline_label,
            variant_label=variant_label,
            comparison_key=comparison_key,
            failed_records=[
                (record, baseline_label if is_baseline(record) else variant_label(record))
                for record in failed_by_category.get(category, [])
            ],
            unmatched_base_results=unmatched_base_by_category.get(category, []),
            unmatched_candidate_results=unmatched_candidate_by_category.get(category, []),
            **table_kwargs,
        )
        for category in (
            set(matches_by_category) | set(failed_by_category)
            | set(unmatched_base_by_category) | set(unmatched_candidate_by_category)
        )
    }

    return assemble_plots_in_grid(
        plots,
        rows={"category": ["linear", "tree-based", "clustering"]},
        columns={"method": ["fit", "predict"]},
        details_by_row=details_by_category,
        notes_by_row=notes_by_row,
    )


def render_preprocessing_views(render_grid: Callable[..., str], results, failed_records) -> str:
    """Both views behind the preprocessing toggle. `render_grid(results,
    failed_records)` renders one of them."""
    default_html = render_grid(results, failed_records)
    preprocessing_html = render_grid(
        add_preprocessing_time([res for res in results if is_real_dataset(res)]),
        [record for record in failed_records if is_real_dataset(record)],
    )
    return (
        '<div class="speedup-view-switch">'
        f'<div class="speedup-view speedup-view-default">{default_html}</div>'
        f'<div class="speedup-view speedup-view-preprocessing">{preprocessing_html}</div>'
        "</div>"
    )


def baseline_view_html(baseline_label: str, *, active: bool, html: str) -> str:
    active_class = " active" if active else ""
    return (
        f'<div class="baseline-view{active_class}" data-baseline="{escape(baseline_label)}">'
        f"{html}</div>"
    )


def baseline_pick_html(label: str, color: str) -> str:
    return (
        '<button type="button" class="baseline-pick" '
        f'data-baseline="{escape(label)}" style="--variant-color: {color}">'
        "Pick as baseline</button>"
    )


def baseline_switch_html(rows: list[str]) -> str:
    """One outer tab: its `rows` must hold the baseline-view and baseline-pick
    elements of `BASELINE_SWITCH_SCRIPT`."""
    return (
        '<div class="baseline-switch">'
        + "".join(f'<div class="page-row">{row}</div>' for row in rows)
        + "</div>"
    )
