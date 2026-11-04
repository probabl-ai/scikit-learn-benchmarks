"""Any software on any machine against either one software on every machine, or
every software on one machine.

The number of combinations grows much faster than the results, so the page holds
none of them: it loads one small data file per (software, machine) variant, and
pairs results and draws the plots in the browser (`html/cross_comparison.js`).
"""
from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path
import shutil

from dashboards import HARDWARE_NAMES
from dashboards import gen_hardware_comparisons as hardware
from sklbench.reporting.comparison import PREPROCESSING_TOGGLE_HTML, tree_binning_note
from sklbench.reporting.envs import read_env, summarize_hardware_env, summarize_software_env
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    HARDWARE_TEMPLATE,
    PLOT_TEMPLATE,
    SOFTWARE_TEMPLATE,
    SOURCE_CELL_TEMPLATE,
    case_hover_html,
    export_columns,
    export_failed_row,
    export_result_row,
    failed_hover_html,
    n_iter_median,
    variant_color_map,
)
from sklbench.reporting.html.glossary import expand_terms
from sklbench.reporting.html.plot import ESTIMATOR_LABEL_ABBREVIATIONS
from sklbench.reporting.html.table import DATASET_KIND_FILTERS, SCORE_METRICS
from sklbench.reporting.matching import (
    CPU_FALLBACK_WARNING,
    CUML_RF_BINS_WARNING,
    METRIC_ABS_TOLERANCE_FLOOR,
    METRIC_REL_TOLERANCE_FLOOR,
    METRIC_STD_TOLERANCE_FACTOR,
    SKLEARNEX_BINS_WARNING,
    BenchmarkRecord,
    Match,
    MethodResult,
    BINNED_SPLIT_WARNING_KINDS,
    is_cuml_random_forest,
    append_cpu_fallback_warning,
    fitted_solver,
    is_real_dataset,
    sklearnex_tree_is_binned,
)
from sklbench.reporting.provenance import summarize_sources


ABOUT_HTML = """<section class="panel">
  <p>This dashboard changes software and hardware at once, to answer questions
  like "is cuML on this GPU faster than scikit-learn-intelex on my server?". A
  <dfn>speed-up</dfn> here mixes both effects. To separate them, see the
  <a href="per_hardware.html">software</a> and
  <a href="hardware_comparisons.html">hardware</a> dashboards.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Pick a <dfn>baseline</dfn>: one software on one machine. Then compare it with
    either one software on every machine (one color per machine), or one machine
    with every software (one color per software). The tabs below describe the
    baseline and each compared machine and <dfn>software environment</dfn>.</p>
    <p>The plots read like in
    <a href="per_hardware.html">the software/implementations dashboard</a>: each
    point is one benchmark case, and points above the dashed 1x line are faster
    than the baseline. Marker shapes mean the same.</p>
    <p><b>Random forests</b> and <b>extra trees</b> often have a different forest
    size on each machine: their times are rescaled to the same size, and their
    metrics aren't compared when the sizes differ. The last row compares the time
    per fit in hyper-parameter searches, where many fits run in parallel.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li>Software and hardware gains can compound. <b>Random forests</b> fit ~30x
      faster with <b>scikit-learn-intelex</b> on the Intel Xeon server than with
      <b>scikit-learn</b> on the Intel Core Ultra laptop, while the software alone
      gives ~2x to 3x and the server alone ~3x.</li>
      <li><b>cuML</b> on the NVIDIA L4 fits <b>LogisticRegression</b> and
      <b>RandomForestClassifier</b> ~3x faster than <b>scikit-learn</b> on the laptop.
      <b>RandomForestRegressor</b> is close to parity and <b>Ridge</b> is often slower.</li>
      <li>A big CPU server can beat that GPU: <b>scikit-learn-intelex</b> on the
      Intel Xeon server fits <b>random forests</b> over 10x faster than <b>cuML</b>
      on the L4. Against the laptop, <b>scikit-learn-intelex</b> and <b>cuML</b> are
      close on most <b>random forests</b>.</li>
      <li><b><dfn>Array API</dfn></b> on a GPU is mostly a <b>LogisticRegression</b> story:
      <b><dfn>PyTorch</dfn></b> on the L4 or the Arc B580 fits it ~10x to 20x faster than
      <b>scikit-learn</b> on the laptop, but <b>Ridge</b> is slower.</li>
    </ul>
  </details>
</section>"""

TITLE = "Cross software/hardware comparison"
SOURCE_CONFIGS = hardware.SOURCE_CONFIGS
SOURCE_ENVS = hardware.SOURCE_ENVS
ENTRY_PAGE = "cross_comparison.html"
DATA_DIR = "cross_comparison_data"
# Stock scikit-learn on the first machine against everything the GPU VM runs,
# cuML included.
DEFAULT_BASELINE_SOFTWARE = "sklearn-pypi"
DEFAULT_TARGET = "hardware-d189c4"


@dataclass(frozen=True)
class Variant:
    software: str
    hardware_hash: str

    @property
    def key(self) -> str:
        return f"{self.software}__{self.hardware_hash}"


def _variant_of(result: MethodResult | BenchmarkRecord) -> Variant:
    return Variant(hardware.software_label(result), result.hardware_hash)


def _group_by_variant(results) -> dict[Variant, list]:
    grouped: dict[Variant, list] = {}
    for result in results:
        grouped.setdefault(_variant_of(result), []).append(result)
    return grouped


def _comparable_targets(results_by_variant, softwares, hardware_hashes) -> dict[str, list[str]]:
    """Baseline key -> keys of the targets it shares at least one case with."""
    match_keys = {
        variant: {
            hardware.match_key(result)
            for result in results
            if result.method in ("fit", "predict")
        }
        for variant, results in results_by_variant.items()
    }
    targets = [
        *((f"software-{software}", lambda v, s=software: v.software == s) for software in softwares),
        *((f"hardware-{h}", lambda v, h=h: v.hardware_hash == h) for h in hardware_hashes),
    ]
    pages = {}
    for software in softwares:
        for hardware_hash in hardware_hashes:
            baseline = Variant(software, hardware_hash)
            if baseline not in match_keys:
                continue
            comparable = [
                key
                for key, includes in targets
                if any(
                    includes(variant) and variant != baseline and match_keys[baseline] & keys
                    for variant, keys in match_keys.items()
                )
            ]
            if comparable:
                pages[baseline.key] = comparable
    return pages


def _compact_times(times: list[float]) -> list[float]:
    return [float(f"{time:.5g}") for time in times]


def _method_block(result: MethodResult) -> dict:
    return {
        "times": _compact_times(result.times),
        "hover": case_hover_html(result.case, result.data_desc),
        "binned": sklearnex_tree_is_binned(result),
        "samples": result.data_desc.get("samples"),
        "features": result.data_desc.get("features"),
    }


def _result_entries(results: list[MethodResult], variant: Variant, case_ids: dict[str, int]) -> list[dict]:
    """One entry per benchmark record, with what `cross_comparison.js` needs to
    pair it with another record and to display it."""
    by_record: dict[int, list[MethodResult]] = {}
    for result in results:
        by_record.setdefault(id(result.record), []).append(result)

    entries = []
    for record_results in by_record.values():
        by_method = {result.method: result for result in record_results}
        first = by_method.get("fit") or next(iter(by_method.values()))
        case_id = case_ids.setdefault(hardware.comparison_key(first), len(case_ids) + 1)
        cpu_fallback = []
        append_cpu_fallback_warning(first, cpu_fallback)
        estimator_params = first.case.get("algorithm", {}).get("estimator_params", {})
        entry = {
            "id": case_id,
            "variant": variant.key,
            "path": first.record.record_path.as_posix(),
            "estimator": first.case.get("algorithm", {}).get("estimator", "unknown"),
            "category": first.category,
            "library": first.implementation.library,
            "real": is_real_dataset(first),
            "sklearnexTree": first.is_sklearnex_tree,
            "cumlRf": is_cuml_random_forest(first),
            "fallback": first.is_sklearnex_fallback,
            "cpuFallback": bool(cpu_fallback),
            "maxBins": estimator_params.get("max_bins"),
            "nEstimators": estimator_params.get("n_estimators"),
            "solver": fitted_solver(first.case, first.implementation.library, first.attributes),
            "nIter": first.attributes.get("n_iter", []),
            "nIterMedian": n_iter_median(first),
            "metrics": Match.comparable_metrics(first.metrics),
            "methods": {
                method: _method_block(by_method[method])
                for method in ("fit", "predict")
                if method in by_method
            },
            "row": export_result_row(first, case_id),
        }
        if "preprocessing" in by_method:
            entry["preprocessingTimes"] = _compact_times(by_method["preprocessing"].times)
        entries.append({name: value for name, value in entry.items() if value is not None})
    return entries


def _failed_entries(records: list[BenchmarkRecord], variant: Variant, case_ids: dict[str, int]) -> list[dict]:
    entries = []
    for record in records:
        case_id = case_ids.setdefault(hardware.comparison_key(record), len(case_ids) + 1)
        entries.append({
            "id": case_id,
            "variant": variant.key,
            "path": record.record_path.as_posix(),
            "estimator": record.case.get("algorithm", {}).get("estimator", "unknown"),
            "category": record.category,
            "real": is_real_dataset(record),
            "binned": sklearnex_tree_is_binned(record),
            "hover": failed_hover_html(record),
            "row": export_failed_row(record, case_id),
        })
    return entries


def _search_entries(results: list[MethodResult], variant: Variant, case_ids: dict[str, int]) -> list[dict]:
    entries = []
    for result in results:
        case_id = case_ids.setdefault(hardware.search_match_key(result), len(case_ids) + 1)
        entries.append({
            "id": case_id,
            "variant": variant.key,
            "path": result.record.record_path.as_posix(),
            "estimator": result.case.get("algorithm", {}).get("estimator", "unknown"),
            "category": hardware.SEARCH_ROW,
            "searchNJobs": result.case["hptuning"].get("n_jobs", 1),
            "nIter": [],
            "metrics": {},
            "methods": {"fit": _method_block(result)},
            "row": export_result_row(result, case_id),
        })
    return entries


def _env_html(result: MethodResult) -> str:
    summary = summarize_software_env(
        read_env("software", result.software_hash),
        result.implementation,
        software_hash=result.software_hash,
    )
    summary["name"] = hardware.software_label(result)
    return (
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", result.hardware_hash)))
        + SOFTWARE_TEMPLATE.render(**summary)
    )


def _source_rows(results: list[MethodResult]) -> list[dict]:
    records = {id(result.record): result.record for result in results}
    return [
        {
            "cell": SOURCE_CELL_TEMPLATE.render(source=row["source"]),
            "hardware": row["hardware"],
            "count": row["count"],
            "start": row["start"].isoformat(),
            "end": row["end"].isoformat(),
            "envs": row["envs"],
            "configs": row["configs"],
        }
        for row in summarize_sources(records.values())
    ]


def _variant_data(variant, results, failed_records, search_results, case_ids) -> dict:
    timestamps = [result.timestamp_recorded for result in results]
    return {
        "envHtml": _env_html(results[0]),
        "count": len(results),
        "start": min(timestamps).isoformat(),
        "end": max(timestamps).isoformat(),
        "sources": _source_rows(results),
        "results": _result_entries(results, variant, case_ids),
        "failed": _failed_entries(failed_records, variant, case_ids),
        "search": _search_entries(search_results, variant, case_ids),
    }


def _write_data_files(output_dir: Path, results_by_variant, failed_by_variant, search_by_variant) -> None:
    data_dir = output_dir / DATA_DIR
    # Variants can disappear from one run to the next.
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)
    # Shared across files: equal ids mean the same case on any two variants.
    case_ids: dict[str, int] = {}
    for variant, results in results_by_variant.items():
        data = _variant_data(
            variant,
            results,
            failed_by_variant.get(variant, []),
            search_by_variant.get(variant, []),
            case_ids,
        )
        payload = json.dumps(data, separators=(",", ":"), default=str)
        (data_dir / f"{variant.key}.js").write_text(
            f'window.sklbenchCrossData["{variant.key}"]={payload};\n'
        )


def _warning_json(warning) -> dict:
    return {
        "icon": warning.icon,
        "short_message": warning.short_message,
        "message": warning.message,
        "kind": warning.kind,
    }


def _page_config(results_by_variant, softwares, hardware_hashes, results) -> dict:
    pages = _comparable_targets(results_by_variant, softwares, hardware_hashes)
    default_baseline = next(
        (key for key in pages if key.startswith(f"{DEFAULT_BASELINE_SOFTWARE}__")),
        next(iter(pages)),
    )
    return {
        "dataDir": DATA_DIR,
        "pages": pages,
        "defaultBaseline": default_baseline,
        "defaultTarget": DEFAULT_TARGET,
        "variants": {
            variant.key: {
                "software": variant.software,
                "hardwareHash": variant.hardware_hash,
                "hardwareLabel": hardware.hardware_label(variant_results[0]),
            }
            for variant, variant_results in results_by_variant.items()
        },
        "machineNames": {h: HARDWARE_NAMES[h] for h in hardware_hashes},
        "labelOrder": [*softwares, *hardware.HARDWARE_ORDER],
        "hardwareColors": hardware.hardware_colors(results),
        "softwareColors": variant_color_map(softwares),
        "warnings": {
            "sklearnexBins": _warning_json(SKLEARNEX_BINS_WARNING),
            "cumlRfBins": _warning_json(CUML_RF_BINS_WARNING),
            "cpuFallback": _warning_json(CPU_FALLBACK_WARNING),
        },
        "binnedWarningKinds": sorted(BINNED_SPLIT_WARNING_KINDS),
        "metricTolerance": {
            "stdFactor": METRIC_STD_TOLERANCE_FACTOR,
            "absFloor": METRIC_ABS_TOLERANCE_FLOOR,
            "relFloor": METRIC_REL_TOLERANCE_FLOOR,
        },
        "scoreMetrics": SCORE_METRICS,
        "datasetKindFilters": DATASET_KIND_FILTERS,
        "columns": export_columns(),
        "estimatorAbbreviations": list(ESTIMATOR_LABEL_ABBREVIATIONS.items()),
        "notes": {
            "tree-based": {
                noun: expand_terms(tree_binning_note(noun)) for noun in ("machine", "software")
            },
        },
        "searchNote": expand_terms(hardware.SEARCH_NOTE),
        "plotTemplate": PLOT_TEMPLATE.to_plotly_json(),
    }


SELECTOR_HTML = """<section class="panel">
  <div class="comparison-selector">
    <label>Baseline software<select id="cross-software"></select></label>
    <span class="comparison-selector-sep">on</span>
    <label>Baseline machine<select id="cross-machine"></select></label>
    <span class="comparison-selector-sep">vs</span>
    <label>Compare with<select id="cross-target"></select></label>
  </div>
</section>"""


def generate(output_dir: Path) -> None:
    results, failed_records = hardware.read_results()
    softwares = sorted({_variant_of(res).software for res in results}, key=hardware.software_sort_key)
    hardware_hashes = [
        hardware_hash
        for hardware_hash in HARDWARE_NAMES
        if any(res.hardware_hash == hardware_hash for res in results)
    ]
    results_by_variant = _group_by_variant(results)
    _write_data_files(
        output_dir,
        results_by_variant,
        _group_by_variant(failed_records),
        _group_by_variant(hardware.read_kept_search_results()),
    )

    config = _page_config(results_by_variant, softwares, hardware_hashes, results)
    script = files("sklbench.reporting.html").joinpath("cross_comparison.js").read_text(encoding="utf-8")
    html = BASE_TEMPLATE.render(title=TITLE, rows=[
        ABOUT_HTML,
        SELECTOR_HTML,
        PREPROCESSING_TOGGLE_HTML,
        '<div id="cross-content"></div>',
        '<script type="application/json" id="cross-config">'
        + json.dumps(config, separators=(",", ":")).replace("</", "<\\/")
        + "</script>",
        f"<script>{script}</script>",
    ])
    output = output_dir / ENTRY_PAGE
    output.write_text(html)
    print(f"Dashboard written to {output}")
