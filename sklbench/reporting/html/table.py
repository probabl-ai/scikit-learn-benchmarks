from __future__ import annotations

import itertools
import json
import math
from statistics import median
from typing import Callable

from ..envs import json_viewer_url, profile_viewer_url
from ..matching import Match, MethodResult
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


def _dataset_name(result: MethodResult) -> str:
    data = result.case.get("data", {})
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


def _result_params(result: MethodResult) -> dict:
    return result.case.get("algorithm", {}).get("estimator_params", {}) or {}


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
        "dataset": _dataset_name(result),
        "variant": variant,
        "n_samples": data_desc.get("samples"),
        "n_features": data_desc.get("features"),
        "fit_time": None,
        "fit_speedup": None,
        "predict_time": None,
        "predict_speedup": None,
        "py_spy_url": _profile_url(result),
        "json_url": _record_json_url(result),
        "hyperparams": _result_params(result),
    }
    return row


def _speedup(base_result: MethodResult, result: MethodResult) -> float | None:
    result_time = median(result.times)
    if result_time == 0:
        return math.inf
    return median(base_result.times) / result_time


def _add_result_method(
    rows: dict[str, dict],
    *,
    result: MethodResult,
    base_result: MethodResult,
    variant: str,
    comparison_key: str,
):
    key = _row_key(result, variant)
    row = rows.setdefault(key, _new_row(result, variant, comparison_key))
    method = result.method
    if method == "fit":
        row["n_samples"] = result.data_desc.get("samples")
        row["n_features"] = result.data_desc.get("features")
    row[f"{method}_time"] = median(result.times)
    row[f"{method}_speedup"] = _speedup(base_result, result)


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
) -> str:
    rows_by_key: dict[str, dict] = {}
    hyperparam_names = set()

    for matches in matches_by_method.values():
        for match in matches:
            base = match.base_result
            result = match.matched_result
            hyperparam_names.update(_result_params(base))
            hyperparam_names.update(_result_params(result))
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
        _column("Variant name", "variant", header_filter=True, sorter="string"),
        _column("n_samples", "n_samples", header_filter=True, sorter="number"),
        _column("n_features", "n_features", header_filter=True, sorter="number"),
    ]
    columns.extend(
        _column(name, field, header_filter=True, sorter="string")
        for name, field in hyperparam_fields.items()
    )
    columns.extend(
        [
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
                "py-spy link",
                "py_spy_url",
                header_sort=False,
                formatter_name="link",
                link_label="py-spy",
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
    return f"""<details class="detailed-results">
  <summary>Detailed results</summary>
  <div class="detailed-results-toolbar" hidden>
    <button id="{reset_button_id}" class="row-filter-reset" type="button" title="Clear row filter" aria-label="Clear row filter">x</button>
  </div>
  <div id="{table_id}" class="detailed-results-table"></div>
  <script>
    document.currentScript.closest("details").addEventListener("toggle", (event) => {{
      if (!event.target.open) {{
        return;
      }}
      sklbenchInitTable(
        "{table_id}",
        {_safe_json(rows)},
        {_safe_json(columns)},
        "{reset_button_id}"
      );
    }}, {{once: true}});
  </script>
</details>"""
