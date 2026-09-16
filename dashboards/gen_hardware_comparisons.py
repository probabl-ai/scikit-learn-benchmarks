"""Hardware A-vs-B comparison, picked interactively from two dropdowns.

Each "hardware variant" is one machine (`HARDWARE_NAMES`/`GPU_NAMES` in
`dashboards/__init__.py`) restricted to either its CPU-device results or its
GPU-device results (`HardwareVariant.devices`) - a machine with a GPU shows up
as two separate options, since "high-end server CPU vs M4 GPU" isn't a
meaningful comparison (different device kinds entirely), while "Intel laptop
GPU vs M4 GPU" is.

For every ordered pair of same-family variants, every build/implementation
variant present on *both* sides (`variant_label`, hardware/device-agnostic -
e.g. "sklearn-pypi", "sklearnex-cpu", "sklearn-torch") is compared against
itself across the two machines, exactly as `variant_label` never mixes across
build/implementation, so a single ratio never conflates a hardware
difference with a software one. Pairs with zero overlapping variant/category/
method matches aren't rendered at all and don't appear as selectable options
- that's the "only propose pairs with matching software results" requirement.

Rendering both directions of every pair (rather than one canonical order plus
a client-side flip) keeps `Match.base_result`/`matched_result` - and so the
speed-up sign and any warnings - correct no matter which side the user picks
as "Baseline": swapping which MethodResult list is `base_results` in
`find_matches` isn't just cosmetic, so it has to actually be recomputed, not
inverted after the fact.
"""
from dataclasses import dataclass, replace
from html import escape
from itertools import permutations
import json

from dashboards import GPU_NAMES, HARDWARE_NAMES, dashboard_output_path
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    assemble_plots_in_grid,
    detailed_results_table_html,
    render_software_tabs,
    speedup_plot_html,
    variant_color_map,
)
from sklbench.reporting.envs import (
    read_env,
    software_build_name,
    summarize_hardware_env,
    summarize_software_env,
)
from sklbench.reporting.matching import (
    append_cpu_fallback_warning,
    append_iterations_warning,
    append_max_bins_warning,
    find_matches,
    read_all_results,
    date_range,
    is_models_scalability_result,
    is_scaling_benchmark,
    BenchmarkRecord,
    Match,
    MatchWarning,
    MethodResult,
)
from sklbench.reporting.utils import stable_json, without_keys


BASE_IMPLEMENTATION = "sklearn"
CATEGORIES = ["linear", "tree-based", "clustering"]
METHODS = ["fit", "predict"]

# A result's `implementation.device` is what separates a machine's CPU
# results from its GPU ones - "None"/"default" and the plain "cpu" tag both
# mean "ran on the CPU" (see e.g. Implementation.short_name), while
# "gpu"/"xpu"/"mps" are the three device tags this repo's configs use across
# sklearnex GPU offload, dpnp/pytorch Array API on an Intel GPU, and pytorch
# Array API on Apple's MPS backend, respectively.
CPU_DEVICES = {None, "default", "cpu"}
GPU_DEVICES = {"gpu", "xpu", "mps"}


@dataclass(frozen=True)
class HardwareVariant:
    key: str
    label: str
    hardware_hash: str
    family: str  # "cpu" or "gpu" - a pairing only ever compares two variants
    # of the same family (see module docstring).

    @property
    def devices(self) -> set[str | None]:
        return CPU_DEVICES if self.family == "cpu" else GPU_DEVICES


HARDWARE_VARIANTS: list[HardwareVariant] = [
    HardwareVariant(f"{hardware_hash}-cpu", name, hardware_hash, "cpu")
    for hardware_hash, name in HARDWARE_NAMES.items()
] + [
    HardwareVariant(f"{hardware_hash}-gpu", name, hardware_hash, "gpu")
    for hardware_hash, name in GPU_NAMES.items()
]
FAMILY_LABELS = {"cpu": "CPU", "gpu": "GPU"}


def variant_results(
    results: list[MethodResult], variant: HardwareVariant
) -> list[MethodResult]:
    return [
        result
        for result in results
        if result.hardware_hash == variant.hardware_hash
        and result.implementation.device in variant.devices
    ]


def variant_label(result: MethodResult | BenchmarkRecord) -> str:
    """The build/implementation identity to hold fixed while comparing
    hardware - deliberately hardware/device-agnostic, since the whole point
    is to pair up the *same* software across two machines whose device tag
    for "the accelerator this ran on" otherwise differs (e.g. Intel GPU's
    Array API pytorch backend is tagged "xpu", Apple's is tagged "mps" - see
    `GPU_DEVICES` - so a raw `Implementation.short_name` would never match
    across those two)."""
    implementation = result.implementation
    if implementation.library == BASE_IMPLEMENTATION and implementation.data_library is None:
        return software_build_name(result.software_hash)
    if implementation.data_library is not None:
        return f"{implementation.library}-{implementation.data_library}"
    return implementation.short_name


# `n_jobs` and RF/ET's `n_estimators` are both derived in `real_datasets.py`
# from `N_JOBS = floor(0.9 * cpu_count(...))` - the *local* machine's core
# count at config-generation time - so they legitimately differ between two
# machines' runs of what's otherwise the identical case. Excluded from the
# match key for the same reason `n_jobs` already is: this comparison should
# still pair these up rather than treat them as different workloads.
_MATCH_EXCLUDED_NAMES = {"implementation", "max_bins", "n_jobs", "n_estimators"}

# Same normalization/target as gen_models_scalability.py's
# `NORMALIZED_N_ESTIMATORS`: RF/ET fit and predict time both scale
# ~linearly with forest size, and since that size differs between machines
# here (see `_MATCH_EXCLUDED_NAMES` above), comparing raw times would
# conflate "faster hardware" with "this pairing happened to fit/predict a
# bigger forest". Rescaling every RF/ET result to a common forest size
# divides that confound back out before speed-ups are computed.
NORMALIZED_N_ESTIMATORS = 100


def _n_estimators(result: MethodResult) -> int | None:
    return result.case.get("algorithm", {}).get("estimator_params", {}).get(
        "n_estimators"
    )


def _normalize_tree_result(result: MethodResult) -> MethodResult:
    n_estimators = _n_estimators(result)
    if not n_estimators:
        return result
    scale = NORMALIZED_N_ESTIMATORS / n_estimators
    return replace(result, times=[t * scale for t in result.times])


# Cross-hardware comparisons in this dashboard routinely pair up results
# whose model isn't actually identical - RF/ET's `n_estimators` varies by
# machine (see `_normalize_tree_result`) - so `Match.metrics_differences`
# and the iteration-count check in `append_iterations_warning` would just
# flag that expected divergence as a reliability warning on every other
# point. Clearing `metrics` and dropping the specific attributes those
# checks read (`has_onedal_estimator`, which also drives
# `is_sklearnex_fallback`'s "fell back to scikit-learn" marker; `n_iter`)
# suppresses that noise while leaving other attributes (e.g. `solver`,
# shown in the detailed table) untouched.
_DROPPED_ATTRIBUTES = {"has_onedal_estimator", "n_iter"}


def _drop_metrics_and_reliability_signals(result: MethodResult) -> MethodResult:
    attributes = {
        name: value
        for name, value in result.attributes.items()
        if name not in _DROPPED_ATTRIBUTES
    }
    return replace(result, metrics={}, attributes=attributes)


def _match_key(result: MethodResult) -> str:
    case = without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES)
    case["method"] = result.method
    return stable_json(case)


def _table_comparison_key(result: MethodResult) -> str:
    return stable_json(without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES))


def result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list[MatchWarning]]:
    assert base_res.hardware_hash != candidate.hardware_hash
    assert variant_label(base_res) == variant_label(candidate)

    warnings = []
    # `append_max_bins_warning` assumes a vanilla-sklearn base compared
    # against a sklearnex candidate - true for e.g. the "sklearn-pypi"/
    # "sklearn-cf-mkl" labels, but for "sklearnex-cpu" both sides are
    # sklearnex (same variant, different hardware), so the warning doesn't
    # apply there.
    if base_res.implementation.library == BASE_IMPLEMENTATION and candidate.is_sklearnex_tree:
        append_max_bins_warning(base_res, candidate, warnings)
    append_iterations_warning(base_res, candidate, warnings)
    # Unlike the other dashboards' base/candidate pairing (always
    # sklearn-vs-accelerated on the same machine), either side here can be
    # the GPU one (e.g. comparing the same "sklearn-torch" variant's MPS
    # results across two machines), so both need the check.
    append_cpu_fallback_warning(base_res, warnings)
    append_cpu_fallback_warning(candidate, warnings)

    return _match_key(base_res) == _match_key(candidate), warnings


def match_variant_label(match: Match) -> str:
    return variant_label(match.matched_result)


def _comparison_page(rows: list[str]) -> str:
    return "".join(f'<div class="page-row">{row}</div>' for row in rows)


def _hardware_env_badge(variant: HardwareVariant) -> str:
    return (
        f"<h3>{escape(variant.label)}</h3>"
        + HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", variant.hardware_hash)))
    )


def _software_env_badges(
    label: str,
    baseline_variant: HardwareVariant,
    candidate_variant: HardwareVariant,
    baseline_results: list[MethodResult],
    candidate_results: list[MethodResult],
) -> list[str]:
    """One badge per side for this build/implementation label - unlike a
    single-hardware-family dashboard, the two sides here can genuinely run a
    different pinned env for the "same" label (e.g. the pytorch version
    backing "sklearn-torch" on an Intel GPU vs on Apple's MPS backend), so
    both are shown rather than picking one as representative."""
    badges = []
    for side_variant, side_results in (
        (baseline_variant, baseline_results),
        (candidate_variant, candidate_results),
    ):
        source = next((r for r in side_results if variant_label(r) == label), None)
        if source is None:
            continue
        summary = summarize_software_env(
            read_env("software", source.software_hash),
            source.implementation,
            software_hash=source.software_hash,
        )
        summary["name"] = f"{label} — {side_variant.label}"
        badges.append(SOFTWARE_TEMPLATE.render(**summary))
    return badges


def render_comparison(
    all_results: list[MethodResult],
    baseline_variant: HardwareVariant,
    candidate_variant: HardwareVariant,
) -> tuple[str, int]:
    """Renders one directional pairing. Returns `(html, match_count)` -
    `match_count == 0` means this pairing has no comparable results at all
    and should neither be rendered nor offered as a dropdown option."""
    baseline_results = [
        _normalize_tree_result(result)
        for result in variant_results(all_results, baseline_variant)
    ]
    candidate_results = [
        _normalize_tree_result(result)
        for result in variant_results(all_results, candidate_variant)
    ]
    empty = ('<section class="empty">No overlapping benchmark results for this pair.</section>', 0)
    if not baseline_results or not candidate_results:
        return empty

    shared_labels = sorted(
        {variant_label(result) for result in baseline_results}
        & {variant_label(result) for result in candidate_results}
    )
    if not shared_labels:
        return empty
    trace_colors = variant_color_map(shared_labels)

    plots = []
    matches_by_category = {}
    total_matches = 0
    for category in CATEGORIES:
        for method in METHODS:
            # One `find_matches` call per label so a baseline result for one
            # variant can never be paired against a candidate result for
            # another - `_match_key` deliberately excludes `implementation`,
            # so that cross-variant mismatch wouldn't otherwise be
            # structurally impossible.
            category_method_matches = [
                match
                for label in shared_labels
                for match in find_matches(
                    [
                        result
                        for result in baseline_results
                        if result.category == category
                        and result.method == method
                        and variant_label(result) == label
                    ],
                    [
                        result
                        for result in candidate_results
                        if result.category == category
                        and result.method == method
                        and variant_label(result) == label
                    ],
                    result_matches,
                    match_key=_match_key,
                )
            ]
            matches_by_category.setdefault(category, {})[method] = category_method_matches
            total_matches += len(category_method_matches)
            plots.append(
                {
                    "category": category,
                    "method": method,
                    "point_count": len(category_method_matches),
                    "plot": speedup_plot_html(
                        category_method_matches,
                        baseline_label=baseline_variant.label,
                        variant_colors=trace_colors,
                        trace_variant=match_variant_label,
                        x_variant=match_variant_label,
                        variant_sort_key=shared_labels.index,
                        comparison_key=_table_comparison_key,
                    ),
                }
            )

    if total_matches == 0:
        return empty

    software_badges = [
        badge
        for label in shared_labels
        for badge in _software_env_badges(
            label, baseline_variant, candidate_variant, baseline_results, candidate_results
        )
    ]

    html = _comparison_page(
        [
            DATE_RANGE_TEMPLATE.render(date_range(baseline_results + candidate_results)),
            render_software_tabs(
                [_hardware_env_badge(baseline_variant), _hardware_env_badge(candidate_variant)]
            ),
            render_software_tabs(software_badges, variant_colors=trace_colors),
            assemble_plots_in_grid(
                plots,
                rows={"category": CATEGORIES},
                columns={"method": METHODS},
                details_by_row={
                    category: detailed_results_table_html(
                        category,
                        category_matches,
                        baseline_label=baseline_variant.label,
                        variant_label=variant_label,
                        comparison_key=_table_comparison_key,
                    )
                    for category, category_matches in matches_by_category.items()
                },
            ),
        ]
    )
    return html, total_matches


def _pair_key(baseline: HardwareVariant, candidate: HardwareVariant) -> str:
    return f"{baseline.key}|{candidate.key}"


def _panel_id(baseline: HardwareVariant, candidate: HardwareVariant) -> str:
    return f"hw-cmp-{baseline.key}-{candidate.key}"


def render_selector(all_results: list[MethodResult]) -> str:
    panels = []
    pair_panels: dict[str, str] = {}
    for baseline, candidate in permutations(HARDWARE_VARIANTS, 2):
        if baseline.family != candidate.family:
            continue
        html, match_count = render_comparison(all_results, baseline, candidate)
        if match_count == 0:
            continue
        panel_id = _panel_id(baseline, candidate)
        panels.append({"id": panel_id, "html": html})
        pair_panels[_pair_key(baseline, candidate)] = panel_id

    if not pair_panels:
        return '<section class="empty">No comparable results across any two hardwares yet.</section>'

    variants_with_options = [
        variant
        for variant in HARDWARE_VARIANTS
        if any(key.startswith(f"{variant.key}|") for key in pair_panels)
    ]
    default_baseline, default_candidate = next(
        (
            (baseline, candidate)
            for baseline, candidate in permutations(HARDWARE_VARIANTS, 2)
            if _pair_key(baseline, candidate) in pair_panels
        )
    )

    panels_html = "".join(
        f'<div id="{panel["id"]}" class="tab-panel">{panel["html"]}</div>' for panel in panels
    )
    variants_json = json.dumps(
        [{"key": v.key, "label": v.label, "family": v.family} for v in variants_with_options]
    )
    pair_panels_json = json.dumps(pair_panels)
    family_labels_json = json.dumps(FAMILY_LABELS)

    return f"""
    <section class="panel hw-compare-controls">
      <div class="hw-compare-row">
        <label>Baseline
          <select id="hw-baseline-select"></select>
        </label>
        <span class="hw-compare-vs">vs</span>
        <label>Compare
          <select id="hw-candidate-select"></select>
        </label>
      </div>
    </section>
    <div class="hw-compare-panels">
      {panels_html}
      <div id="hw-compare-empty" class="tab-panel">
        <section class="empty">No comparable benchmark results for this pair.</section>
      </div>
    </div>
    <script>
    (function() {{
      const variants = {variants_json};
      const pairPanels = {pair_panels_json};
      const familyLabels = {family_labels_json};

      const baselineSelect = document.getElementById("hw-baseline-select");
      const candidateSelect = document.getElementById("hw-candidate-select");

      function candidatesFor(baselineKey) {{
        return variants.filter(
          (v) => v.key !== baselineKey && pairPanels[`${{baselineKey}}|${{v.key}}`]
        );
      }}

      function populate(select, items, selectedKey) {{
        select.innerHTML = "";
        const byFamily = {{}};
        items.forEach((item) => {{
          (byFamily[item.family] = byFamily[item.family] || []).push(item);
        }});
        Object.entries(byFamily).forEach(([family, familyItems]) => {{
          const group = document.createElement("optgroup");
          group.label = familyLabels[family] || family;
          familyItems.forEach((item) => {{
            const option = document.createElement("option");
            option.value = item.key;
            option.textContent = item.label;
            if (item.key === selectedKey) {{
              option.selected = true;
            }}
            group.appendChild(option);
          }});
          select.appendChild(group);
        }});
      }}

      function showPanel() {{
        const baseline = baselineSelect.value;
        const candidate = candidateSelect.value;
        const panelId = pairPanels[`${{baseline}}|${{candidate}}`] || "hw-compare-empty";
        document.querySelectorAll(".hw-compare-panels > .tab-panel").forEach((panel) => {{
          panel.classList.toggle("active", panel.id === panelId);
        }});
        const activePanel = document.getElementById(panelId);
        activePanel.querySelectorAll(".plotly-graph-div").forEach((chart) => {{
          if (chart.offsetParent !== null) {{
            Plotly.Plots.resize(chart);
          }}
        }});
        if (typeof sklbenchWirePlotClicks === "function") {{
          sklbenchWirePlotClicks();
        }}
      }}

      baselineSelect.addEventListener("change", () => {{
        const options = candidatesFor(baselineSelect.value);
        const kept = options.find((o) => o.key === candidateSelect.value);
        populate(candidateSelect, options, kept ? kept.key : options[0] && options[0].key);
        showPanel();
      }});
      candidateSelect.addEventListener("change", showPanel);

      populate(
        baselineSelect,
        variants.filter((v) => candidatesFor(v.key).length > 0),
        "{default_baseline.key}"
      );
      populate(candidateSelect, candidatesFor(baselineSelect.value), "{default_candidate.key}");
      showPanel();
    }})();
    </script>
    """


if __name__ == "__main__":
    all_results = [
        _drop_metrics_and_reliability_signals(result)
        for result in read_all_results()
        if not is_scaling_benchmark(result) and not is_models_scalability_result(result)
    ]

    html = BASE_TEMPLATE.render(
        title="sklbench hardware comparison dashboard",
        rows=[render_selector(all_results)],
    )

    output = dashboard_output_path("hardware_comparisons.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
