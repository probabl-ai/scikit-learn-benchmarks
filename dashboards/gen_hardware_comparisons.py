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
from functools import lru_cache
from html import escape
from itertools import permutations
import json
from pathlib import Path

from dashboards import (
    GPU_NAMES, HARDWARE_NAMES, GENERAL_SOURCE_CONFIGS, GENERAL_SOURCE_ENVS,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
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
    append_solver_warning,
    append_max_bins_warning,
    find_matches,
    read_all_results,
    date_range,
    matches_source_configs,
    BenchmarkRecord,
    Match,
    MatchWarning,
    MethodResult,
)
from sklbench.reporting.utils import stable_json, without_keys


ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares machines running the same software. Every build
  or implementation present on both machines is matched, so a speed-up comes
  from the hardware.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Pick a baseline and a comparison machine. The dropdowns only offer CPU
    vs CPU or GPU vs GPU pairs that have comparable results. Each cell shows
    the fit or predict speed-up (log scale) per estimator category, one line
    per build or implementation.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li>The high-end Intel Xeon server mostly helps <b>random forests</b> and <b>extra
      trees</b>. With <b>scikit-learn</b>, their fit is ~2x to 4x faster than on the
      Intel Ultra laptop. Their predict is close to parity.</li>
      <li><b>scikit-learn-intelex</b> makes much better use of the server's 172
      cores. With it, <b>random forests</b> and <b>extra trees</b> fit ~10x to 15x
      faster on the server than on the laptop.</li>
      <li>For <b>linear models</b>, the server is often slower than the
      laptop. A single fit doesn't use many cores well, so the laptop's
      faster cores probably win.</li>
      <li><b>HistGradientBoosting</b> results are hard to interpret, because
      of known scalability issues on laptops with <b>conda-forge</b> builds and on
      machines with many cores. See
      <a href="https://github.com/scikit-learn/scikit-learn/issues/34764">scikit-learn#34764</a>
      and the <a href="hgb_scaling.html">HistGradientBoosting
      thread-scalability breakdown</a>.</li>
      <li>On the <b>PyPI</b> build, the Apple M4 CPU is ~1.3x to 2x slower
      than the Intel Ultra laptop.</li>
    </ul>
  </details>
</section>"""


BASE_IMPLEMENTATION = "sklearn"
CATEGORIES = ["linear", "tree-based", "clustering"]
METHODS = ["fit", "predict"]

# A result's `implementation.device` is what separates a machine's CPU
# results from its GPU ones - "None"/"default" and the plain "cpu" tag both
# mean "ran on the CPU" (see e.g. Implementation.short_name), while
# "gpu"/"xpu"/"mps"/"cuda" are the device tags this repo's configs use across
# sklearnex GPU offload, dpnp/pytorch Array API on an Intel GPU, pytorch
# Array API on Apple's MPS backend, and pytorch/cupy Array API on an NVIDIA
# GPU, respectively.
CPU_DEVICES = {None, "default", "cpu"}
GPU_DEVICES = {"gpu", "xpu", "mps", "cuda"}

# Unlike CPU, where every build/implementation variant present on both sides
# gets compared, GPU pairings are restricted to the one backend that's
# actually portable across GPU vendors: the Array API pytorch backend
# (data_library "torch"), which runs on Intel's "xpu", Apple's "mps" and
# NVIDIA's "cuda" devices. sklearnex's native GPU offload (oneDAL) and the
# dpnp Array API backend are Intel-only, and cupy is NVIDIA-only, so they're
# excluded from GPU variant results outright rather than left to fall out of
# an incidental label intersection (see `variant_results`).
GPU_PORTABLE_DATA_LIBRARIES = {"torch"}


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
    results = [
        result
        for result in results
        if result.hardware_hash == variant.hardware_hash
        and result.implementation.device in variant.devices
    ]
    if variant.family == "gpu":
        results = [
            result
            for result in results
            if result.implementation.data_library in GPU_PORTABLE_DATA_LIBRARIES
        ]
    return results


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


# `n_jobs` and RF/ET's `n_estimators` are both derived in `_real_datasets.py`
# from `N_JOBS = floor(0.9 * cpu_count(...))` - the *local* machine's core
# count at config-generation time - so they legitimately differ between two
# machines' runs of what's otherwise the identical case. Excluded from the
# match key for the same reason `n_jobs` already is: this comparison should
# still pair these up rather than treat them as different workloads.
# `source_config` is excluded for the same reason as in
# `MethodResult.minimal_match_key`.
_MATCH_EXCLUDED_NAMES = {
    "implementation", "max_bins", "n_jobs", "n_estimators", "source_config",
}

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
# would just flag that expected divergence as a reliability warning on every
# other point. Clearing `metrics` and dropping `has_onedal_estimator` (which
# also drives `is_sklearnex_fallback`'s "fell back to scikit-learn" marker)
# suppresses that noise while leaving other attributes (e.g. `solver`,
# `n_iter`, shown in the detailed table) untouched.
_DROPPED_ATTRIBUTES = {"has_onedal_estimator"}


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
    # No `append_iterations_warning`: n_iter is only shown in the detailed
    # table here, not flagged on the plots.
    append_solver_warning(base_res, candidate, warnings)
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


def _vs_table(
    baseline_variant: HardwareVariant,
    candidate_variant: HardwareVariant,
    rows: list[tuple[str, str, str]],
    uncompared: frozenset[str] = frozenset(),
) -> str:
    """`rows` are `(name, baseline_html, candidate_html)`. A row whose two
    sides are identical collapses into one merged cell so that the
    highlighted, split rows are exactly the differences. `uncompared` rows
    (e.g. per-side links) are always split and never highlighted."""
    body = []
    for name, baseline_html, candidate_html in rows:
        if name in uncompared:
            cells = f"<td>{baseline_html}</td><td>{candidate_html}</td>"
        elif baseline_html == candidate_html:
            cells = f'<td class="same" colspan="2">{baseline_html}</td>'
        else:
            cells = (
                f'<td class="differs">{baseline_html}</td>'
                f'<td class="differs">{candidate_html}</td>'
            )
        body.append(f"<tr><th scope=\"row\">{escape(name)}</th>{cells}</tr>")
    return (
        '<table class="vs-table"><thead><tr><th></th>'
        f"<th>Baseline: {escape(baseline_variant.label)}</th>"
        f"<th>Comparison: {escape(candidate_variant.label)}</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _lines(items: list[str]) -> str:
    return "<br>".join(escape(item) for item in items) or '<span class="muted">none</span>'


def _hardware_rows(summary: dict) -> dict[str, str]:
    price = summary["price_label"] or "unknown"
    gpus = []
    for gpu in summary["gpus"]:
        detail = f"{gpu['memory_gb']} GB"
        if gpu["integrated"]:
            detail += ", integrated"
        gpus.append(f"{gpu['name']} ({detail})")
    return {
        "CPU": escape(str(summary["cpu_name"])),
        "Architecture": escape(str(summary["architecture"])),
        "Cores": escape(
            f"{summary['physical_cores']} physical, {summary['logical_cpus']} logical"
        ),
        "RAM": escape(f"{summary['ram_gb']} GB"),
        "Price": escape(price),
        "GPU(s)": _lines(gpus),
    }


def _hardware_panel(
    baseline_variant: HardwareVariant, candidate_variant: HardwareVariant
) -> str:
    baseline, candidate = (
        _hardware_rows(summarize_hardware_env(read_env("hardware", variant.hardware_hash)))
        for variant in (baseline_variant, candidate_variant)
    )
    rows = [(name, baseline[name], candidate[name]) for name in baseline]
    return (
        '<section class="panel"><h2>Hardware</h2>'
        + _vs_table(baseline_variant, candidate_variant, rows)
        + "</section>"
    )


def _software_rows(summary: dict) -> dict[str, str]:
    rows = {"Python": escape(summary["python_version"])}
    for package in summary["packages"]:
        cell = escape(str(package["version"]))
        if package["kind"]:
            cell += f' <span class="muted">({escape(package["kind"])})</span>'
        rows[package["name"]] = cell
    # threadpoolctl reports libraries in load order, which isn't stable
    # across machines.
    rows["Threadpools"] = _lines(sorted(summary["threadpools"]))
    rows["OpenMP"] = _lines(summary["openmp"])
    rows["Full environment"] = (
        f'<a href="{escape(summary["software_env_json_url"])}">view pixi env JSON</a>'
    )
    return rows


def _software_panel(
    label: str,
    baseline_variant: HardwareVariant,
    candidate_variant: HardwareVariant,
    baseline_results: list[MethodResult],
    candidate_results: list[MethodResult],
) -> str:
    """Both sides are shown because the "same" label can still run a
    different pinned env on each machine (e.g. the pytorch version backing
    "sklearn-torch" on an Intel GPU vs on Apple's MPS backend)."""
    sides = []
    for side_results in (baseline_results, candidate_results):
        source = next(r for r in side_results if variant_label(r) == label)
        sides.append(
            _software_rows(
                summarize_software_env(
                    read_env("software", source.software_hash),
                    source.implementation,
                    software_hash=source.software_hash,
                )
            )
        )
    baseline, candidate = sides
    missing = '<span class="muted">not installed</span>'
    names = list(baseline) + [name for name in candidate if name not in baseline]
    rows = [
        (name, baseline.get(name, missing), candidate.get(name, missing))
        for name in names
    ]
    return _vs_table(
        baseline_variant, candidate_variant, rows, uncompared=frozenset({"Full environment"})
    )


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
                    "case_count": len(
                        {
                            _table_comparison_key(match.base_result)
                            for match in category_method_matches
                        }
                    ),
                    "plot": speedup_plot_html(
                        category_method_matches,
                        baseline_label=baseline_variant.label,
                        y_title=f"{candidate_variant.label} speed-up",
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
    hardware_labels = {
        baseline_variant.hardware_hash: baseline_variant.label,
        candidate_variant.hardware_hash: candidate_variant.label,
    }

    software_panels = [
        _software_panel(
            label, baseline_variant, candidate_variant, baseline_results, candidate_results
        )
        for label in shared_labels
    ]

    html = _comparison_page(
        [
            DATE_RANGE_TEMPLATE.render(date_range(baseline_results + candidate_results)),
            _hardware_panel(baseline_variant, candidate_variant),
            render_software_tabs(
                software_panels, variant_colors=trace_colors, labels=shared_labels
            ),
            assemble_plots_in_grid(
                plots,
                rows={"category": CATEGORIES},
                columns={"method": METHODS},
                details_by_row={
                    category: detailed_results_table_html(
                        category,
                        category_matches,
                        baseline_label=variant_label,
                        variant_label=variant_label,
                        variant_column_title="Implementation",
                        hardware_label=lambda result: hardware_labels[result.hardware_hash],
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

      function showPanel(updateHash = true) {{
        const baseline = baselineSelect.value;
        const candidate = candidateSelect.value;
        if (updateHash) {{
          const params = new URLSearchParams({{baseline, compare: candidate}});
          history.replaceState(null, "", `#${{params}}`);
        }}
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
      candidateSelect.addEventListener("change", () => showPanel());

      const hashParams = new URLSearchParams(window.location.hash.slice(1));
      let initialBaseline = "{default_baseline.key}";
      let initialCandidate = "{default_candidate.key}";
      if (pairPanels[`${{hashParams.get("baseline")}}|${{hashParams.get("compare")}}`]) {{
        initialBaseline = hashParams.get("baseline");
        initialCandidate = hashParams.get("compare");
      }}
      populate(
        baselineSelect,
        variants.filter((v) => candidatesFor(v.key).length > 0),
        initialBaseline
      );
      populate(candidateSelect, candidatesFor(baselineSelect.value), initialCandidate);
      // Don't rewrite the URL on load, so an unmodified default page keeps a clean URL.
      showPanel(false);
    }})();
    </script>
    """


SOURCE_CONFIGS = GENERAL_SOURCE_CONFIGS
# The conda-forge BLAS/OpenMP build variants are a software comparison
# (gen_builds_comparison.py); across hardware they'd only multiply the lines
# per cell, so sklearn-cf-mkl alone stands in for the conda-forge builds.
SOURCE_ENVS = [
    env
    for env in GENERAL_SOURCE_ENVS
    if not env.startswith("sklearn-cf-") or env == "sklearn-cf-mkl"
]


@lru_cache(maxsize=None)
def _pixi_env_name(software_hash: str) -> str:
    return read_env("software", software_hash)["pixi_environment_name"]


def _is_excluded_env(result: MethodResult) -> bool:
    name = _pixi_env_name(result.software_hash)
    return name.startswith("sklearn-cf-") and name not in SOURCE_ENVS


def generate(output_dir: Path) -> None:
    all_results = [
        _drop_metrics_and_reliability_signals(result)
        for result in read_all_results()
        if matches_source_configs(result.case, SOURCE_CONFIGS)
        and not _is_excluded_env(result)
    ]

    html = BASE_TEMPLATE.render(
        title="sklbench hardware comparison dashboard",
        rows=[ABOUT_HTML, render_selector(all_results)],
    )

    output = output_dir / "hardware_comparisons.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
