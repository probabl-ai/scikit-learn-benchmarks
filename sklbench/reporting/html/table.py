from __future__ import annotations

import itertools
import json
import math
from statistics import median
from typing import Callable

from ..envs import json_viewer_url, profile_viewer_url
from ..matching import BenchmarkRecord, Match, MethodResult
from ..utils import stable_json, without_keys


table_ids = itertools.count()


def _safe_json(value):
    return json.dumps(value, sort_keys=True, default=str).replace("</", "<\\/")


def _format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _dataset_name(case: dict) -> str:
    data = case.get("data", {})
    return data.get("dataset") or data.get("source") or "unknown"


def _row_key(result: MethodResult, variant: str) -> str:
    record = result.record
    if record is not None and record.record_path is not None:
        return f"{variant}:{record.record_path.as_posix()}"

    case = without_keys(result.case, excluded_names={"method"})
    return stable_json(
        {
            "variant": variant,
            "case": case,
            "hardware": result.hardware_hash,
            "software": result.software_hash,
            "timestamp": result.timestamp_recorded.isoformat(),
        }
    )


def _record_json_url(result: MethodResult) -> str | None:
    record = result.record
    if record is None or record.record_path is None:
        return None
    return json_viewer_url(record.record_path)


def _profile_url(result: MethodResult) -> str | None:
    record = result.record
    if record is None or record.profile_path is None:
        return None
    return profile_viewer_url(record.profile_path)


def _profile_label(record: BenchmarkRecord | None) -> str | None:
    if record is None or record.profile_path is None:
        return None
    # `.prof.gz` comes from the cProfile pass (`_run_cprofile_pass` in
    # sklbench/orchestrator/implementation.py); `.raw.gz`/`.svg` come from py-spy.
    if record.profile_path.name.endswith(".prof.gz"):
        return "cProfile"
    return "py-spy"


def _result_params(case: dict) -> dict:
    return case.get("algorithm", {}).get("estimator_params", {}) or {}


def _default_comparison_key(result: MethodResult) -> str:
    return stable_json(
        without_keys(result.case, excluded_names={"implementation", "max_bins"})
    )


def _new_row(
    result: MethodResult,
    variant: str,
    comparison_key: str,
) -> dict:
    data_desc = result.data_desc or {}
    row = {
        "comparison_key": comparison_key,
        "estimator": result.case.get("algorithm", {}).get("estimator", "unknown"),
        "dataset": _dataset_name(result.case),
        "variant": variant,
        "n_samples": data_desc.get("samples"),
        "n_features": data_desc.get("features"),
        "fit_time": None,
        "fit_speedup": None,
        "predict_time": None,
        "predict_speedup": None,
        "profile_url": _profile_url(result),
        "profile_url_label": _profile_label(result.record),
        "json_url": _record_json_url(result),
        "hyperparams": _result_params(result.case),
        "status": "ok",
    }
    return row


def _failed_status(failed_case: dict) -> str:
    return "timed out" if failed_case.get("return_code") == -9 else "failed"


def _failed_row_key(record: BenchmarkRecord, variant: str) -> str:
    if record.record_path is not None:
        return f"{variant}:{record.record_path.as_posix()}"
    case = without_keys(record.case, excluded_names={"method"})
    return stable_json(
        {
            "variant": variant,
            "case": case,
            "hardware": record.hardware_hash,
            "software": record.software_hash,
            "timestamp": record.timestamp_recorded.isoformat(),
        }
    )


def _new_failed_row(
    record: BenchmarkRecord,
    variant: str,
    comparison_key: str,
) -> dict:
    generation_kwargs = record.case.get("data", {}).get("generation_kwargs", {})
    failed_case = record.failed_case or {}
    return {
        "comparison_key": comparison_key,
        "estimator": record.case.get("algorithm", {}).get("estimator", "unknown"),
        "dataset": _dataset_name(record.case),
        "variant": variant,
        "n_samples": generation_kwargs.get("n_samples"),
        "n_features": generation_kwargs.get("n_features"),
        "fit_time": None,
        "fit_speedup": None,
        "predict_time": None,
        "predict_speedup": None,
        "profile_url": None,
        "json_url": json_viewer_url(record.record_path) if record.record_path else None,
        "hyperparams": _result_params(record.case),
        "status": _failed_status(failed_case),
    }


def _speedup(base_result: MethodResult, result: MethodResult) -> float | None:
    result_time = median(result.times)
    if result_time == 0:
        return math.inf
    return median(base_result.times) / result_time


def _add_result_method(
    rows: dict[str, dict],
    *,
    result: MethodResult,
    variant: str,
    comparison_key: str,
    base_result: MethodResult | None = None,
):
    key = _row_key(result, variant)
    row = rows.setdefault(key, _new_row(result, variant, comparison_key))
    method = result.method
    if method == "fit":
        row["n_samples"] = result.data_desc.get("samples")
        row["n_features"] = result.data_desc.get("features")
    row[f"{method}_time"] = median(result.times)
    row[f"{method}_speedup"] = (
        _speedup(base_result, result) if base_result is not None else None
    )


def _column(
    title: str,
    field: str,
    *,
    visible: bool | None = None,
    header_filter: bool = False,
    header_sort: bool | None = None,
    sorter: str | None = None,
    formatter_name: str | None = None,
    link_label: str | None = None,
) -> dict:
    column = {"title": title, "field": field}
    if visible is not None:
        column["visible"] = visible
    if header_filter:
        column["headerFilter"] = "list"
        column["headerFilterParams"] = {
            "clearable": True,
            "sort": "asc",
            "valuesLookup": True,
        }
    if header_sort is not None:
        column["headerSort"] = header_sort
    if sorter is not None:
        column["sorter"] = sorter
    if formatter_name is not None:
        column["formatterName"] = formatter_name
    if link_label is not None:
        column["linkLabel"] = link_label
    return column


def detailed_results_table_html(
    category: str,
    matches_by_method: dict[str, list[Match]],
    *,
    baseline_label: str,
    variant_label: Callable[[MethodResult], str],
    comparison_key: Callable[[MethodResult], str] = _default_comparison_key,
    failed_records: list[tuple[BenchmarkRecord, str]] = (),
    unmatched_base_results: list[MethodResult] = (),
    unmatched_candidate_results: list[MethodResult] = (),
    open: bool = False,
    variant_column_title: str = "Variant name",
    default_variant_filter: str | None = None,
) -> str:
    rows_by_key: dict[str, dict] = {}
    hyperparam_names = set()

    for matches in matches_by_method.values():
        for match in matches:
            base = match.base_result
            result = match.matched_result
            hyperparam_names.update(_result_params(base.case))
            hyperparam_names.update(_result_params(result.case))
            _add_result_method(
                rows_by_key,
                result=base,
                base_result=base,
                variant=baseline_label,
                comparison_key=comparison_key(base),
            )
            _add_result_method(
                rows_by_key,
                result=result,
                base_result=base,
                variant=variant_label(result),
                comparison_key=comparison_key(result),
            )

    for record, variant in failed_records:
        hyperparam_names.update(_result_params(record.case))
        key = _failed_row_key(record, variant)
        rows_by_key[key] = _new_failed_row(
            record, variant, comparison_key(record)
        )

    # Results whose counterpart failed never appear in `matches_by_method`
    # (find_matches only pairs up results that both succeeded) - add them here so
    # they still show up next to the failed row they'd otherwise have matched.
    # No speedup is computable for either side, since there's no successful
    # counterpart to compare against.
    for result in unmatched_base_results:
        hyperparam_names.update(_result_params(result.case))
        _add_result_method(
            rows_by_key,
            result=result,
            variant=baseline_label,
            comparison_key=comparison_key(result),
        )

    for result in unmatched_candidate_results:
        hyperparam_names.update(_result_params(result.case))
        _add_result_method(
            rows_by_key,
            result=result,
            variant=variant_label(result),
            comparison_key=comparison_key(result),
        )

    if not rows_by_key:
        return ""

    row_hyperparams = []
    varying_hyperparam_names = []
    for row in rows_by_key.values():
        row_hyperparams.append(row["hyperparams"])

    for name in sorted(hyperparam_names):
        values = {
            _format_value(hyperparams.get(name))
            for hyperparams in row_hyperparams
        }
        if len(values) > 1:
            varying_hyperparam_names.append(name)

    hyperparam_fields = {
        name: f"hp_{index}" for index, name in enumerate(varying_hyperparam_names)
    }
    rows = []
    for row in rows_by_key.values():
        hyperparams = row.pop("hyperparams")
        for name, field in hyperparam_fields.items():
            row[field] = _format_value(hyperparams.get(name))
        rows.append(row)

    rows = sorted(
        rows,
        key=lambda row: (
            row["estimator"],
            row["dataset"],
            row["variant"],
            row.get("n_samples") or -1,
            row.get("n_features") or -1,
        ),
    )

    columns = [
        _column("comparison_key", "comparison_key", visible=False, header_sort=False),
        _column("Estimator name", "estimator", header_filter=True, sorter="string"),
        _column("Dataset name", "dataset", header_filter=True, sorter="string"),
        _column(variant_column_title, "variant", header_filter=True, sorter="string"),
        _column("n_samples", "n_samples", header_filter=True, sorter="number"),
        _column("n_features", "n_features", header_filter=True, sorter="number"),
    ]
    columns.extend(
        _column(name, field, header_filter=True, sorter="string")
        for name, field in hyperparam_fields.items()
    )
    columns.extend(
        [
            _column("Status", "status", header_filter=True, sorter="string"),
            _column("fit time", "fit_time", sorter="number", formatter_name="duration"),
            _column(
                "fit speed up",
                "fit_speedup",
                sorter="number",
                formatter_name="speedup",
            ),
            _column(
                "predict time",
                "predict_time",
                sorter="number",
                formatter_name="duration",
            ),
            _column(
                "predict speed up",
                "predict_speedup",
                sorter="number",
                formatter_name="speedup",
            ),
            _column(
                "profile link",
                "profile_url",
                header_sort=False,
                formatter_name="link",
                link_label="profile",
            ),
            _column(
                "JSON link",
                "json_url",
                header_sort=False,
                formatter_name="link",
                link_label="JSON",
            ),
        ]
    )

    table_id = f"detailed-results-{next(table_ids)}"
    reset_button_id = f"{table_id}-reset"
    default_header_filters = (
        {"variant": default_variant_filter} if default_variant_filter else {}
    )
    init_call = (
        f'sklbenchInitTable("{table_id}", {_safe_json(rows)}, '
        f'{_safe_json(columns)}, "{reset_button_id}", {_safe_json(default_header_filters)});'
    )
    # `<details open>` alone doesn't fire a "toggle" event on page load, so
    # an eagerly-visible table needs the init call to run unconditionally
    # instead of waiting on that event - a real code-path difference, not
    # just a markup attribute.
    script = (
        f"<script>{init_call}</script>"
        if open
        else f"""<script>
    document.currentScript.closest("details").addEventListener("toggle", (event) => {{
      if (!event.target.open) {{
        return;
      }}
      {init_call}
    }}, {{once: true}});
  </script>"""
    )
    return f"""<details class="detailed-results"{" open" if open else ""}>
  <summary>Detailed results</summary>
  <div class="detailed-results-toolbar" hidden>
    <button id="{reset_button_id}" class="row-filter-reset" type="button" title="Clear row filter" aria-label="Clear row filter">x</button>
  </div>
  <div id="{table_id}" class="detailed-results-table"></div>
  {script}
</details>"""
