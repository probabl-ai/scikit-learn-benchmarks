"""Data files and page shell of the software, builds, hardware and cross comparison
dashboards. Their pages hold no results: `html/speedup_comparison.js` loads one
data file per variant on screen, then `html/speedup_core.js` pairs results and
builds plots and tables.
"""

from html import escape
from importlib.resources import files
import json
from pathlib import Path
import shutil
from typing import Callable

from .html import (
    PLOT_TEMPLATE,
    SOURCE_CELL_TEMPLATE,
    case_hover_html,
    export_columns,
    export_failed_row,
    export_result_row,
    failed_hover_html,
    n_iter_median,
)
from .html.glossary import expand_terms
from .html.plot import ESTIMATOR_LABEL_ABBREVIATIONS
from .html.table import DATASET_KIND_FILTERS, SCORE_METRICS
from .matching import (
    BINNED_SPLIT_WARNING_KINDS,
    CPU_FALLBACK_WARNING,
    CUML_RF_BINS_WARNING,
    METRIC_ABS_TOLERANCE_FLOOR,
    METRIC_REL_TOLERANCE_FLOOR,
    METRIC_STD_TOLERANCE_FACTOR,
    SKLEARNEX_BINS_WARNING,
    BenchmarkRecord,
    Match,
    MethodResult,
    append_cpu_fallback_warning,
    fitted_solver,
    is_cuml_random_forest,
    is_real_dataset,
    is_sklearnex_tree,
    uses_binned_splits,
)
from .provenance import summarize_sources
from .utils import stable_json, without_keys


PREPROCESSING_TOGGLE_HTML = (
    '<section class="panel">'
    '<label class="preprocessing-toggle">'
    '<input type="checkbox" class="preprocessing-toggle-checkbox">'
    " Include <dfn>preprocessing</dfn> time in fit speed-ups (real datasets only)"
    "</label>"
    "</section>"
)
CONTENT_HTML = '<div id="comparison-content"></div>'


def tree_binning_note(variant_noun: str) -> str:
    return (
        f"Tree-based plots: within each {variant_noun}, cases without <dfn>binning</dfn> "
        "(<dfn>exact splits</dfn>) are on the left, cases with binning "
        "(<dfn>histogram-based splits</dfn> or <dfn>quantile-based splits</dfn>) "
        "on the right."
    )


# cuML random forests always bin.
CUML_TREE_NOTE = (
    " cuML random forests are only paired with the binned scikit-learn-intelex cases."
)


def baseline_pick_html(label: str, color: str) -> str:
    return (
        '<button type="button" class="baseline-pick" '
        f'data-baseline="{escape(label)}" style="--variant-color: {color}">'
        "Pick as baseline</button>"
    )


def same_machine_case_key(result: MethodResult | BenchmarkRecord) -> str:
    """With the same hardware on both sides, `n_jobs` and forest sizes are part of a case."""
    return stable_json(
        without_keys(result.case, excluded_names={"implementation", "max_bins", "source_config"})
    )


def _compact_times(times: list[float]) -> list[float]:
    return [float(f"{time:.5g}") for time in times]


def _method_block(result: MethodResult) -> dict:
    return {
        "times": _compact_times(result.times),
        "hover": case_hover_html(result.case, result.data_desc),
        "binned": uses_binned_splits(result),
        "samples": result.data_desc.get("samples"),
        "features": result.data_desc.get("features"),
    }


def _result_entries(results: list[MethodResult], variant_key: str, case_id) -> list[dict]:
    """One entry per benchmark record, with what `speedup_comparison.js` needs to
    pair it with another record and to display it."""
    by_record: dict[int, list[MethodResult]] = {}
    for result in results:
        by_record.setdefault(id(result.record), []).append(result)

    entries = []
    for record_results in by_record.values():
        by_method = {result.method: result for result in record_results}
        first = by_method.get("fit") or next(iter(by_method.values()))
        entry_id = case_id(first)
        cpu_fallback = []
        append_cpu_fallback_warning(first, cpu_fallback)
        estimator_params = first.case.get("algorithm", {}).get("estimator_params", {})
        entry = {
            "id": entry_id,
            "variant": variant_key,
            "path": first.record.record_path.as_posix(),
            "timestamp": first.timestamp_recorded.timestamp(),
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
            "row": export_result_row(first, entry_id),
        }
        if "preprocessing" in by_method:
            entry["preprocessingTimes"] = _compact_times(by_method["preprocessing"].times)
        entries.append({name: value for name, value in entry.items() if value is not None})
    return entries


def _failed_entries(records: list[BenchmarkRecord], variant_key: str, case_id) -> list[dict]:
    entries = []
    for record in records:
        entry_id = case_id(record)
        entries.append({
            "id": entry_id,
            "variant": variant_key,
            "path": record.record_path.as_posix(),
            "estimator": record.case.get("algorithm", {}).get("estimator", "unknown"),
            "category": record.category,
            "real": is_real_dataset(record),
            "binned": uses_binned_splits(record),
            "sklearnexTree": is_sklearnex_tree(record),
            "cumlRf": is_cuml_random_forest(record),
            "hover": failed_hover_html(record),
            "row": export_failed_row(record, entry_id),
        })
    return entries


def _search_entries(results: list[MethodResult], variant_key: str, case_id) -> list[dict]:
    entries = []
    for result in results:
        entry_id = case_id(result)
        entries.append({
            "id": entry_id,
            "variant": variant_key,
            "path": result.record.record_path.as_posix(),
            "estimator": result.case.get("algorithm", {}).get("estimator", "unknown"),
            "category": "hyper-parameter search",
            "searchNJobs": result.case["hptuning"].get("n_jobs", 1),
            "nIter": [],
            "metrics": {},
            "methods": {"fit": _method_block(result)},
            "row": export_result_row(result, entry_id),
        })
    return entries


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


def write_data_files(
    output_dir: Path,
    data_dir: str,
    results_by_variant: dict[str, list[MethodResult]],
    *,
    failed_by_variant: dict[str, list[BenchmarkRecord]],
    case_key: Callable[[MethodResult | BenchmarkRecord], str],
    env_html: Callable[[str, MethodResult], str],
    search_by_variant: dict[str, list[MethodResult]] | None = None,
    search_key: Callable[[MethodResult], str] | None = None,
) -> None:
    """Writes one file per variant key. Equal `id`s across files mean the same
    case, so `case_key` is what decides which results pair up."""
    directory = output_dir / data_dir
    # Variants can disappear from one run to the next.
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)

    ids: dict[str, int] = {}

    def case_id(result) -> int:
        return ids.setdefault(case_key(result), len(ids) + 1)

    def search_id(result) -> int:
        return ids.setdefault(f"search:{search_key(result)}", len(ids) + 1)

    for variant_key, results in results_by_variant.items():
        timestamps = [result.timestamp_recorded for result in results]
        data = {
            "envHtml": env_html(variant_key, results[0]),
            "count": len(results),
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
            "sources": _source_rows(results),
            "results": _result_entries(results, variant_key, case_id),
            "failed": _failed_entries(failed_by_variant.get(variant_key, []), variant_key, case_id),
            "search": _search_entries(
                (search_by_variant or {}).get(variant_key, []), variant_key, search_id
            ),
        }
        payload = json.dumps(data, separators=(",", ":"), default=str)
        (directory / f"{variant_key}.js").write_text(
            f"window.sklbenchComparisonData[{json.dumps(variant_key)}]={payload};\n"
        )


def _warning_json(warning) -> dict:
    return {
        "icon": warning.icon,
        "short_message": warning.short_message,
        "message": warning.message,
        "kind": warning.kind,
    }


def page_rows(page_config: dict) -> list[str]:
    """Page rows to add last, after `CONTENT_HTML` and `PREPROCESSING_TOGGLE_HTML`.
    `page_config` holds what is specific to the page."""
    config = {
        **page_config,
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
        "plotTemplate": PLOT_TEMPLATE.to_plotly_json(),
    }
    assets = files("sklbench.reporting.html")
    core = assets.joinpath("speedup_core.js").read_text(encoding="utf-8")
    script = assets.joinpath("speedup_comparison.js").read_text(encoding="utf-8")
    return [
        '<script type="application/json" id="comparison-config">'
        + json.dumps(config, separators=(",", ":")).replace("</", "<\\/")
        + "</script>",
        f"<script>{core}</script>",
        f"<script>{script}</script>",
    ]


def expanded_note(html: str) -> str:
    """Notes end up in the page's JSON config, where `BASE_TEMPLATE` can't expand
    their terms."""
    return expand_terms(html)
