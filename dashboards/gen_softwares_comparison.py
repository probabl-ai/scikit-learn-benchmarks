from html import escape
from pathlib import Path

from dashboards import HARDWARE_NAMES, GENERAL_SOURCE_CONFIGS, GENERAL_SOURCE_ENVS
from sklbench.reporting.utils import (
    partition_iterable, groupby, stable_json, without_keys,
)

from sklbench.reporting.matching import (
    append_iterations_warning, append_solver_warning, append_max_bins_warning, read_all_results,
    read_failed_records, find_matches, date_range, BenchmarkRecord, Match,
    MatchWarning, MethodResult, append_cpu_fallback_warning,
    matches_source_configs,
    add_preprocessing_time, is_real_dataset,
)

from sklbench.reporting.envs import (
    is_vanilla_sklearn, read_env, software_build_name, summarize_software_env,
    summarize_hardware_env,
)
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


BASE_IMPLEMENTATION = "sklearn"
# Alternative scikit-learn builds offered as a baseline but never plotted as
# a candidate: next to every other variant they'd make the plots unreadable.
BASELINE_ONLY_BUILDS = {"sklearn-cf-mkl"}
# base.css's --muted, so these tabs don't look like a plotted variant.
BASELINE_ONLY_TAB_COLOR = "#48566b"
ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares implementations on the same machine: stock
  scikit-learn, scikit-learn-intelex and Array API backends (PyTorch, dpnp,
  CuPy). Since the hardware is fixed, a speed-up comes from the software:
  faster algorithms, better use of the CPU (vectorization, threading), or
  offloading to the GPU.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Pick a machine tab. The top of the tab describes the machine and each
    software environment (versions, BLAS and OpenMP libraries). The baseline
    defaults to stock scikit-learn from PyPI. To change it, open another
    software tab and click "Pick as baseline". The conda-forge MKL build
    (<code>sklearn-cf-mkl</code>) can be picked as a baseline too, but isn't
    plotted otherwise.</p>
    <p>The plots are arranged in a grid: one row per estimator category
    (linear, tree-based, clustering) and one column for <code>fit</code> and
    one for <code>predict</code>. In each plot:</p>
    <ul>
      <li>the x-axis lists the estimators, with one color per implementation.
      An implementation keeps its color on every machine and for every
      baseline;</li>
      <li>each point is one benchmark case (dataset and hyperparameters);</li>
      <li>the y-axis is the speed-up over the baseline, in log scale. Points
      above the dashed 1x line are faster than the baseline.</li>
    </ul>
    <p>The marker shape flags cases to double check:</p>
    <ul>
      <li>circle (●): metrics and setup match the baseline;</li>
      <li>square (■): metrics match but the setup differs, for instance a
      different solver or number of iterations, or scikit-learn-intelex using
      histogram-based splits where scikit-learn uses exact ones;</li>
      <li>open diamond (◇): the metrics differ;</li>
      <li>grey point (<span style="color: grey">●</span>):
      scikit-learn-intelex fell back to scikit-learn;</li>
      <li>cross (✕): failed run.</li>
    </ul>
    <p>The notes under each plot count these cases.</p>
    <p>Hover a point to see its case. Click it to find its row in "Detailed
    results", the table below each row, which has the exact timings and
    metrics and can be filtered by column.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li><code>sklearnex-cpu</code> is the <b>most consistently fast option</b>.
      Linear models gain less than tree-based models and KMeans.</li>
      <li>sklearnex tree-based models are <b>fast, but don't behave exactly
      like scikit-learn</b>:
        <ul>
          <li>By default, sklearnex random forests and extra trees <b>bin
          features</b> (<code>max_bins=256</code>), which scikit-learn doesn't
          implement yet. Binning <b>roughly doubles or triples the fit
          speed-up</b>. Cases run with exact splits
          (<code>max_bins=n_samples</code>) are on the left of each
          implementation in the tree-based plots.</li>
          <li>Two <b>oneDAL bugs</b> make some fits worse:
          <a href="https://github.com/uxlfoundation/oneDAL/issues/3648">oneDAL#3648</a>
          (ExtraTreesRegressor) and
          <a href="https://github.com/uxlfoundation/oneDAL/issues/3771">oneDAL#3771</a>
          (<code>max_leaf_nodes</code>).</li>
        </ul>
      </li>
      <li>sklearnex KMeans <b>fits ~3x faster</b> and is never slower than
      scikit-learn.</li>
      <li>Array API backends are <b>mixed</b>. Only LogisticRegression and
      Ridge are benchmarked for now:
        <ul>
          <li>LogisticRegression on GPU is <b>fairly fast</b>.</li>
          <li>Ridge uses the SVD solver instead of Cholesky under Array API,
          which is <b>probably why it's slower</b>.</li>
          <li>PyTorch on CPU parallelizes every operation, including small
          vector operations. That's why LogisticRegression is slower than
          scikit-learn with NumPy.</li>
        </ul>
      </li>
    </ul>
  </details>
</section>"""
PREPROCESSING_TOGGLE_HTML = (
    '<section class="panel">'
    '<label class="preprocessing-toggle">'
    '<input type="checkbox" class="preprocessing-toggle-checkbox">'
    " Include preprocessing time in fit speed-ups (real datasets only)"
    "</label>"
    "</section>"
)
TREE_BINNING_NOTE = (
    "Tree-based plots: within each implementation, cases without binning "
    "(exact splits) are on the left, cases with binning (histogram-based "
    "splits) on the right."
)


def is_alt_sklearn_build(result: MethodResult | BenchmarkRecord) -> bool:
    return (
        result.implementation.short_name == BASE_IMPLEMENTATION
        and not is_vanilla_sklearn(result.software_hash)
    )


def is_shown(result: MethodResult | BenchmarkRecord) -> bool:
    return not is_alt_sklearn_build(result) or variant_label(result) in BASELINE_ONLY_BUILDS


def is_plotted(result: MethodResult | BenchmarkRecord) -> bool:
    return not is_alt_sklearn_build(result)


def variant_label(result: MethodResult | BenchmarkRecord) -> str:
    if result.implementation.short_name == BASE_IMPLEMENTATION:
        return software_build_name(result.software_hash)
    return result.implementation.short_name


def _variant_sort_key(result: MethodResult | BenchmarkRecord) -> tuple[int, str]:
    if result.implementation.short_name == BASE_IMPLEMENTATION:
        rank = 1 if is_alt_sklearn_build(result) else 0
    elif result.implementation.library == "sklearnex":
        rank = 2
    else:  # Array API backends
        rank = 3
    return rank, variant_label(result)


def sorted_variant_labels(results: list[MethodResult]) -> list[str]:
    """Stock scikit-learn first, then its alternative builds, sklearnex and
    Array API backends."""
    return [
        variant_label(res)
        for res in sorted(
            {variant_label(res): res for res in results}.values(),
            key=_variant_sort_key,
        )
    ]


def _max_bins(result: MethodResult) -> int | None:
    return result.case.get("algorithm", {}).get("estimator_params", {}).get("max_bins")


def _case_key(case: dict) -> str:
    """Case identity ignoring implementation/max_bins - shared by a base result
    and the candidate(s) it would be compared against (or vice versa)."""
    return stable_json(without_keys(case, excluded_names={"implementation", "max_bins"}))


def result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list[MatchWarning]]:
    """Assumes both results ran on the same hardware with different variants.
    Either side can be sklearnex or run on a GPU, depending on the picked
    baseline."""
    assert base_res.hardware_hash == candidate.hardware_hash
    assert variant_label(base_res) != variant_label(candidate)

    warnings = []

    if base_res.is_sklearnex_tree and candidate.is_sklearnex_tree:
        # sklearnex tree cases are run both with the default max_bins and
        # with exact splits: pair each with its counterpart.
        if _max_bins(base_res) != _max_bins(candidate):
            return False, warnings
    elif candidate.is_sklearnex_tree:
        append_max_bins_warning(base_res, candidate, warnings)
    elif base_res.is_sklearnex_tree and not base_res.is_sklearnex_fallback:
        append_max_bins_warning(candidate, base_res, warnings)
    append_iterations_warning(base_res, candidate, warnings)
    append_solver_warning(base_res, candidate, warnings)
    append_cpu_fallback_warning(base_res, warnings)
    append_cpu_fallback_warning(candidate, warnings)

    return (
        base_res.minimal_match_key == candidate.minimal_match_key,
        warnings
    )


def _render_speedup_grid(
    results: list[MethodResult],
    failed_records: list[BenchmarkRecord],
    *,
    baseline_label: str,
    variant_colors: dict[str, str],
) -> str:
    """Renders the category x fit/predict speed-up grid for one baseline and
    one view of the page (the default view, or the "with preprocessing" view
    - real datasets only, preprocessing time folded into fit)."""
    def is_baseline(result: MethodResult | BenchmarkRecord) -> bool:
        return variant_label(result) == baseline_label

    results = [res for res in results if is_plotted(res) or is_baseline(res)]
    failed_records = [
        record for record in failed_records if is_plotted(record) or is_baseline(record)
    ]
    base_results, other_results = partition_iterable(results, predicate=is_baseline)
    if not base_results:
        return f'<section class="empty">No {baseline_label} baseline results for this hardware.</section>'

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
        # A sklearnex baseline has one tree result per max_bins value for the
        # same case. `find_matches` keeps a single baseline per case, so
        # match each max_bins value separately.
        base_by_max_bins: dict[int | None, list[MethodResult]] = {}
        for base in group_base_results:
            base_by_max_bins.setdefault(_max_bins(base), []).append(base)
        matches = [
            match
            for max_bins_base_results in base_by_max_bins.values()
            for match in find_matches(max_bins_base_results, other_results, result_matches)
        ]
        matches_by_category.setdefault(category, {})[method] = matches
        # create a JS snippet for plotly:
        plots.append({
            "category": category,
            "method": method,
            "point_count": len(matches),
            "plot": speedup_plot_html(
                matches,
                baseline_label=baseline_label,
                variant_colors=variant_colors,
                trace_variant=lambda match: variant_label(match.matched_result),
                failed_records=candidate_failed_by_category.get(category, []),
            )
        })
    failed_by_category = groupby(failed_records, lambda record: record.category)

    # A failed record means find_matches never sees a pair for that case, so the
    # side that *did* succeed - the base when a candidate failed, or any
    # candidate when the base itself failed - would otherwise silently vanish
    # from the table too. Look those up by case identity so they still show up
    # (with no speedup, since there's nothing successful to compare against).
    base_by_case_key: dict[str, list[MethodResult]] = {}
    for base in base_results:
        base_by_case_key.setdefault(_case_key(base.case), []).append(base)
    other_by_case_key: dict[str, list[MethodResult]] = {}
    for other in other_results:
        other_by_case_key.setdefault(_case_key(other.case), []).append(other)

    unmatched_base_by_category: dict[str, list[MethodResult]] = {}
    unmatched_candidate_by_category: dict[str, list[MethodResult]] = {}
    for record in failed_records:
        key = _case_key(record.case)
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
            failed_records=[
                (record, variant_label(record))
                for record in failed_by_category.get(category, [])
            ],
            unmatched_base_results=unmatched_base_by_category.get(category, []),
            unmatched_candidate_results=unmatched_candidate_by_category.get(category, []),
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
        notes_by_row={"tree-based": TREE_BINNING_NOTE},
    )


def render_hardware_page(
    results: list[MethodResult],
    failed_records: list[BenchmarkRecord],
    hardware_hash: str,
    *,
    variant_colors: dict[str, str],
) -> str | None:
    results = [res for res in results if res.hardware_hash == hardware_hash]
    results = [res for res in results if is_shown(res)]
    failed_records = [
        record for record in failed_records
        if record.hardware_hash == hardware_hash and is_shown(record)
    ]
    if not results:
        return None
    hardwares_set = {res.hardware_hash for res in results}
    if len(hardwares_set) > 1:
        raise ValueError(f"Results are dirty: several hardware hashes match {hardware_hash!r}")

    labels = sorted_variant_labels(results)
    if len(sorted_variant_labels([res for res in results if is_plotted(res)])) < 2:
        return None
    default_baseline = labels[0]

    preprocessing_results = add_preprocessing_time(
        [res for res in results if is_real_dataset(res)]
    )
    preprocessing_failed_records = [
        record for record in failed_records if is_real_dataset(record)
    ]
    baseline_views = []
    for baseline_label in labels:
        default_grid_html = _render_speedup_grid(
            results, failed_records,
            baseline_label=baseline_label, variant_colors=variant_colors,
        )
        preprocessing_grid_html = _render_speedup_grid(
            preprocessing_results, preprocessing_failed_records,
            baseline_label=baseline_label, variant_colors=variant_colors,
        )
        active = " active" if baseline_label == default_baseline else ""
        baseline_views.append(
            f'<div class="baseline-view{active}" data-baseline="{escape(baseline_label)}">'
            '<div class="speedup-view-switch">'
            f'<div class="speedup-view speedup-view-default">{default_grid_html}</div>'
            f'<div class="speedup-view speedup-view-preprocessing">{preprocessing_grid_html}</div>'
            "</div></div>"
        )
    speedup_views_html = "".join(baseline_views)

    hardware_hash, = hardwares_set
    hardware_env = read_env("hardware", hardware_hash)

    tab_colors = {
        **variant_colors,
        **{label: BASELINE_ONLY_TAB_COLOR for label in BASELINE_ONLY_BUILDS},
    }
    first_result_by_label = {}
    for res in results:
        first_result_by_label.setdefault(variant_label(res), res)
    softwares = []
    for label in labels:
        res = first_result_by_label[label]
        summary = summarize_software_env(
            read_env("software", res.software_hash),
            res.implementation,
            software_hash=res.software_hash,
        )
        summary["name"] = label
        summary["actions_html"] = (
            '<button type="button" class="baseline-pick" '
            f'data-baseline="{escape(label)}" style="--variant-color: {tab_colors[label]}">'
            "Pick as baseline</button>"
        )
        softwares.append(summary)

    rows = [
        DATE_RANGE_TEMPLATE.render(date_range(results)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(hardware_env)),
        render_software_tabs([
            SOFTWARE_TEMPLATE.render(**summary)
            for summary in softwares
        ], variant_colors=tab_colors, labels=labels),
        speedup_views_html,
    ]
    return (
        '<div class="baseline-switch">'
        + "".join(f'<div class="page-row">{row}</div>' for row in rows)
        + "</div>"
    )


# The picked baseline is kept across machine tabs when that machine has it.
BASELINE_SWITCH_SCRIPT = """<script>
function sklbenchShowBaseline(baselineSwitch, baseline) {
  baselineSwitch.querySelectorAll(".baseline-view").forEach((view) => {
    view.classList.toggle("active", view.dataset.baseline === baseline);
  });
  baselineSwitch.querySelectorAll(".baseline-pick").forEach((button) => {
    const picked = button.dataset.baseline === baseline;
    button.disabled = picked;
    button.textContent = picked ? "\u2713 Current baseline" : "Pick as baseline";
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


SOURCE_CONFIGS = GENERAL_SOURCE_CONFIGS
SOURCE_ENVS = GENERAL_SOURCE_ENVS


def generate(output_dir: Path) -> None:
    results = [
        res for res in read_all_results()
        if matches_source_configs(res.case, SOURCE_CONFIGS)
    ]
    failed_records = [
        record for record in read_failed_records()
        if matches_source_configs(record.case, SOURCE_CONFIGS)
    ]
    hardware_hashes_with_results = {res.hardware_hash for res in results}
    # Computed over every machine so a variant keeps its color across tabs.
    variant_colors = variant_color_map(
        sorted_variant_labels([res for res in results if is_plotted(res)])
    )
    hardware_pages = [
        (
            hardware_name,
            render_hardware_page(
                results, failed_records, hardware_hash, variant_colors=variant_colors
            ),
        )
        for hardware_hash, hardware_name in HARDWARE_NAMES.items()
        if hardware_hash in hardware_hashes_with_results
    ]

    html = BASE_TEMPLATE.render(rows=[
        ABOUT_HTML,
        PREPROCESSING_TOGGLE_HTML,
        render_hardware_tabs(hardware_pages),
        BASELINE_SWITCH_SCRIPT,
    ])

    output = output_dir / "per_hardware.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
