from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
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


def _record_json_url(
    result: MethodResult, json_url_fn: Callable[[Path], str | None]
) -> str | None:
    record = result.record
    if record is None or record.record_path is None:
        return None
    return json_url_fn(record.record_path)


def _profile_url(
    result: MethodResult, profile_url_fn: Callable[[Path], str | None]
) -> str | None:
    record = result.record
    if record is None or record.profile_path is None:
        return None
    return profile_url_fn(record.profile_path)


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


# Model params shown in the detailed results table, beyond this the table gets
# too wide to be useful. "solver" is overwritten with the fitted
# `estimator.solver_` (see `_row_hyperparams`) rather than the requested param,
# since solvers are often auto-selected.
HYPERPARAM_DISPLAY_ALLOWLIST = ["solver", "n_estimators", "n_clusters"]


def _row_hyperparams(case: dict, attributes: dict | None = None) -> dict:
    params = _result_params(case)
    attributes = attributes or {}
    hyperparams = {}
    for name in HYPERPARAM_DISPLAY_ALLOWLIST:
        if name == "solver":
            solver_values = attributes.get("solver")
            if solver_values:
                hyperparams["solver"] = solver_values[0]
                continue
        if name in params:
            hyperparams[name] = params[name]
    return hyperparams


def default_comparison_key(result: MethodResult) -> str:
    """Shared with `speedup_plot_html` (as its `comparison_key` param) so
    speed-up plot points and detailed-results table rows for the same
    case agree on the key used to link a plot click to its table row."""
    return stable_json(
        without_keys(result.case, excluded_names={"implementation", "max_bins"})
    )


def _row_columns_kind(case: dict) -> str | None:
    """The synthetic tree datasets' column-type mix (e.g. "mix", "binary",
    "continuous", "long-tail") - only set in generation_kwargs for
    tree-based configs (see configs/synthetic_trees.py, hgb_scaling.py)."""
    return case.get("data", {}).get("generation_kwargs", {}).get("columns")


def _row_max_bins(case: dict, library: str, n_samples: int | None) -> str | None:
    """sklearnex's max_bins setting for tree-based results: "default" (not
    overridden - sklearnex's own default of 255) or "n_samples" (explicitly
    set equal to n_samples, i.e. exact/unbinned splits - see
    configs/synthetic_trees.py and append_max_bins_warning in matching.py).
    Empty for sklearn, which doesn't vary this param in these benchmarks."""
    if library != "sklearnex":
        return None
    estimator_params = case.get("algorithm", {}).get("estimator_params", {})
    if "max_bins" not in estimator_params:
        return "default"
    if n_samples is not None and estimator_params["max_bins"] == n_samples:
        return "n_samples"
    return "default"


def _new_row(
    result: MethodResult,
    variant: str,
    comparison_key: str,
    json_url_fn: Callable[[Path], str | None],
    profile_url_fn: Callable[[Path], str | None],
) -> dict:
    data_desc = result.data_desc or {}
    n_samples = result.case.get("data", {}).get("generation_kwargs", {}).get(
        "n_samples"
    )
    if n_samples is None:
        n_samples = data_desc.get("samples")
    row = {
        "comparison_key": comparison_key,
        "estimator": result.case.get("algorithm", {}).get("estimator", "unknown"),
        "dataset": _dataset_name(result.case),
        "variant": variant,
        "n_samples": data_desc.get("samples"),
        "n_features": data_desc.get("features"),
        "columns": _row_columns_kind(result.case),
        "max_bins": _row_max_bins(
            result.case, result.implementation.library, n_samples
        ),
        "fit_time": None,
        "fit_speedup": None,
        "predict_time": None,
        "predict_speedup": None,
        "profile_url": _profile_url(result, profile_url_fn),
        "profile_url_label": _profile_label(result.record),
        "json_url": _record_json_url(result, json_url_fn),
        "hyperparams": _row_hyperparams(result.case, result.attributes),
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
    json_url_fn: Callable[[Path], str | None],
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
        "columns": generation_kwargs.get("columns"),
        "max_bins": _row_max_bins(
            record.case, record.implementation.library, generation_kwargs.get("n_samples")
        ),
        "fit_time": None,
        "fit_speedup": None,
        "predict_time": None,
        "predict_speedup": None,
        "profile_url": None,
        "json_url": json_url_fn(record.record_path) if record.record_path else None,
        "hyperparams": _row_hyperparams(record.case),
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
    json_url_fn: Callable[[Path], str | None],
    profile_url_fn: Callable[[Path], str | None],
    base_result: MethodResult | None = None,
):
    key = _row_key(result, variant)
    row = rows.setdefault(
        key, _new_row(result, variant, comparison_key, json_url_fn, profile_url_fn)
    )
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
    baseline_label: str | Callable[[MethodResult], str],
    variant_label: Callable[[MethodResult], str],
    comparison_key: Callable[[MethodResult], str] = default_comparison_key,
    failed_records: list[tuple[BenchmarkRecord, str]] = (),
    unmatched_base_results: list[MethodResult] = (),
    unmatched_candidate_results: list[MethodResult] = (),
    open: bool = False,
    variant_column_title: str = "Variant name",
    default_variant_filter: str | None = None,
    json_url_fn: Callable[[Path], str | None] = json_viewer_url,
    profile_url_fn: Callable[[Path], str | None] = profile_viewer_url,
) -> str:
    rows_by_key: dict[str, dict] = {}
    hyperparam_names = set(HYPERPARAM_DISPLAY_ALLOWLIST)

    resolve_baseline_label = (
        baseline_label if callable(baseline_label) else (lambda _result: baseline_label)
    )

    for matches in matches_by_method.values():
        for match in matches:
            base = match.base_result
            result = match.matched_result
            _add_result_method(
                rows_by_key,
                result=base,
                base_result=base,
                variant=resolve_baseline_label(base),
                comparison_key=comparison_key(base),
                json_url_fn=json_url_fn,
                profile_url_fn=profile_url_fn,
            )
            _add_result_method(
                rows_by_key,
                result=result,
                base_result=base,
                variant=variant_label(result),
                comparison_key=comparison_key(result),
                json_url_fn=json_url_fn,
                profile_url_fn=profile_url_fn,
            )

    for record, variant in failed_records:
        key = _failed_row_key(record, variant)
        rows_by_key[key] = _new_failed_row(
            record, variant, comparison_key(record), json_url_fn
        )

    # Results whose counterpart failed never appear in `matches_by_method`
    # (find_matches only pairs up results that both succeeded) - add them here so
    # they still show up next to the failed row they'd otherwise have matched.
    # No speedup is computable for either side, since there's no successful
    # counterpart to compare against.
    for result in unmatched_base_results:
        _add_result_method(
            rows_by_key,
            result=result,
            variant=resolve_baseline_label(result),
            comparison_key=comparison_key(result),
            json_url_fn=json_url_fn,
            profile_url_fn=profile_url_fn,
        )

    for result in unmatched_candidate_results:
        _add_result_method(
            rows_by_key,
            result=result,
            variant=variant_label(result),
            comparison_key=comparison_key(result),
            json_url_fn=json_url_fn,
            profile_url_fn=profile_url_fn,
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
        _column(variant_column_title, "variant", header_filter=True, sorter="string"),
        _column("Estimator name", "estimator", header_filter=True, sorter="string"),
        _column("Dataset name", "dataset", header_filter=True, sorter="string"),
        _column("n_samples", "n_samples", header_filter=True, sorter="number"),
        _column("n_features", "n_features", header_filter=True, sorter="number"),
    ]
    if any(row.get("columns") for row in rows):
        columns.append(
            _column("columns", "columns", header_filter=True, sorter="string")
        )
    if any(row.get("max_bins") for row in rows):
        columns.append(
            _column("max_bins", "max_bins", header_filter=True, sorter="string")
        )
    columns.extend(
        _column(name, field, header_filter=True, sorter="string")
        for name, field in hyperparam_fields.items()
    )
    if any(row["status"] != "ok" for row in rows):
        columns.append(
            _column("Status", "status", header_filter=True, sorter="string")
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
        ]
    )
    if any(row["profile_url"] for row in rows):
        columns.append(
            _column(
                "profile link",
                "profile_url",
                header_sort=False,
                formatter_name="link",
                link_label="profile",
            )
        )
    columns.append(
        _column(
            "JSON link",
            "json_url",
            header_sort=False,
            formatter_name="link",
            link_label="JSON",
        )
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
    <button id="{reset_button_id}" class="row-filter-reset" type="button" title="Clear row sort" aria-label="Clear row sort">x</button>
  </div>
  <div id="{table_id}" class="detailed-results-table"></div>
  {script}
</details>"""
