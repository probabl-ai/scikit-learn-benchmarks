from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

from ..envs import (
    effective_openmp_value,
    json_viewer_url,
    openmp_runtime_short_label,
    profile_viewer_url,
)
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


def _failed_status(failed_case: dict) -> str:
    return "timed out" if failed_case.get("return_code") == -9 else "failed"


@dataclass(frozen=True)
class RowInputs:
    """Normalizes a live `MethodResult` and a failed `BenchmarkRecord` into
    one shape, so every `ColumnSpec`/`ColumnGroupSpec` getter below is
    written once and works for both (see `_method_result_inputs` /
    `_failed_record_inputs`)."""

    case: dict
    software_hash: str
    library: str
    data_desc: dict
    attributes: dict
    status: str


def _method_result_inputs(result: MethodResult) -> RowInputs:
    return RowInputs(
        case=result.case,
        software_hash=result.software_hash,
        library=result.implementation.library,
        data_desc=result.data_desc or {},
        attributes=result.attributes or {},
        status="ok",
    )


def _failed_record_inputs(record: BenchmarkRecord) -> RowInputs:
    generation_kwargs = record.case.get("data", {}).get("generation_kwargs", {}) or {}
    return RowInputs(
        case=record.case,
        software_hash=record.software_hash,
        library=record.implementation.library,
        # No run ever happened, so there's no measured data_desc - the
        # config's own declared shape is the closest equivalent, and matches
        # what a successful run of the same case would have reported.
        data_desc={
            "samples": generation_kwargs.get("n_samples"),
            "features": generation_kwargs.get("n_features"),
        },
        attributes={},
        status=_failed_status(record.failed_case or {}),
    )


def _result_params(case: dict) -> dict:
    return case.get("algorithm", {}).get("estimator_params", {}) or {}


# Model params shown in the detailed results table, beyond this the table gets
# too wide to be useful. "solver" is overwritten with the fitted
# `estimator.solver_` (see `_row_hyperparams`) rather than the requested param,
# since solvers are often auto-selected.
HYPERPARAM_DISPLAY_ALLOWLIST = ["solver", "n_estimators", "n_clusters"]


def _row_hyperparams(inputs: RowInputs) -> dict:
    params = _result_params(inputs.case)
    hyperparams = {}
    for name in HYPERPARAM_DISPLAY_ALLOWLIST:
        if name == "solver":
            solver_values = inputs.attributes.get("solver")
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


def _row_columns_kind(inputs: RowInputs) -> str | None:
    """The synthetic tree datasets' column-type mix (e.g. "mix", "binary",
    "continuous", "long-tail") - only set in generation_kwargs for
    tree-based configs (see configs/synthetic_trees.py, hgb_scalability.py)."""
    return inputs.case.get("data", {}).get("generation_kwargs", {}).get("columns")


def _row_order(inputs: RowInputs) -> str | None:
    """The data's memory layout ("C" or "F"): the config-forced value where a
    config varies it (see configs/synthetic_linear.py's `order` field), else
    the measured layout of the loaded array (real datasets - see
    sklbench/runners/datasets/__init__.py's `_measure_order`)."""
    order = inputs.case.get("data", {}).get("order")
    if order is not None:
        return order
    return inputs.data_desc.get("order")


def _row_env(inputs: RowInputs) -> dict:
    """The `bench.env` vars this case ran with (e.g. a BLAS thread-count
    sweep via OMP_NUM_THREADS/OPENBLAS_NUM_THREADS - see
    configs/all_models_logistic_lbfgs_only.py) - kept under a top-level "env"
    key by `_case_without_bench` in matching.py, since the rest of "bench" is
    stripped from case identity before it ever reaches this table."""
    return inputs.case.get("env", {}) or {}


def _row_openmp(inputs: RowInputs) -> str:
    """Short "libgomp"/"libomp" label for the OpenMP runtime this row's build
    links against - a build property rather than a case one, but only
    interesting once more than one OpenMP runtime shows up in the same
    table (e.g. comparing a `sklearn-dev` build against `sklearn-dev-libomp`
    - see configs/_implementations.py)."""
    return openmp_runtime_short_label(inputs.software_hash)


def _row_gomp_spincount(inputs: RowInputs) -> str | None:
    return effective_openmp_value(
        inputs.case.get("env"), inputs.software_hash, "GOMP_SPINCOUNT"
    )


def _row_kmp_blocktime(inputs: RowInputs) -> str | None:
    return effective_openmp_value(
        inputs.case.get("env"), inputs.software_hash, "KMP_BLOCKTIME"
    )


def _row_n_samples_for_max_bins(inputs: RowInputs) -> int | None:
    declared = (
        inputs.case.get("data", {}).get("generation_kwargs", {}).get("n_samples")
    )
    return declared if declared is not None else inputs.data_desc.get("samples")


def _row_max_bins(inputs: RowInputs) -> str | None:
    """sklearnex's max_bins setting for tree-based results: "default" (not
    overridden - sklearnex's own default of 255) or "n_samples" (explicitly
    set equal to n_samples, i.e. exact/unbinned splits - see
    configs/synthetic_trees.py and append_max_bins_warning in matching.py).
    Empty for sklearn, which doesn't vary this param in these benchmarks."""
    if inputs.library != "sklearnex":
        return None
    estimator_params = inputs.case.get("algorithm", {}).get("estimator_params", {})
    if "max_bins" not in estimator_params:
        return "default"
    n_samples = _row_n_samples_for_max_bins(inputs)
    if n_samples is not None and estimator_params["max_bins"] == n_samples:
        return "n_samples"
    return "default"


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the detailed-results table: `getter` extracts its raw
    value from a row's normalized `RowInputs` (works uniformly for a live
    result or a failed record). `show_if_varies` gates whether the column is
    included at all: when True, it's only added once at least two distinct
    (formatted) values appear across all rows - a column that's constant
    for a whole table rarely earns its width. When False, the column always
    shows once declared.

    To add a case/software-derived column for a specific dashboard, add a
    `ColumnSpec` to `COLUMNS` below - no changes needed anywhere else in this
    module. A column needing external, call-time context instead (a URL
    builder, per-method fit/predict state...) doesn't fit this shape; see the
    bespoke fields built directly in `_new_row`/`detailed_results_table_html`."""

    title: str
    field: str
    getter: Callable[[RowInputs], Any]
    show_if_varies: bool = False
    sorter: str = "string"


COLUMNS: list[ColumnSpec] = [
    ColumnSpec("columns", "columns", _row_columns_kind, show_if_varies=True),
    ColumnSpec("order", "order", _row_order, show_if_varies=True),
    ColumnSpec("max_bins", "max_bins", _row_max_bins, show_if_varies=True),
    ColumnSpec("OpenMP", "openmp", _row_openmp, show_if_varies=True),
    ColumnSpec(
        "GOMP_SPINCOUNT", "gomp_spincount", _row_gomp_spincount, show_if_varies=True
    ),
    ColumnSpec(
        "KMP_BLOCKTIME", "kmp_blocktime", _row_kmp_blocktime, show_if_varies=True
    ),
]


@dataclass(frozen=True)
class ColumnGroupSpec:
    """One family of dynamically-named columns, one per key that varies
    across rows (e.g. one column per hyperparameter, one per env var).
    `getter` returns this row's whole `{key: raw_value}` dict; `allowed_keys`
    restricts which keys are even considered (None = every key seen across
    all rows). Every generated column is shown only where it varies - same
    reasoning as `ColumnSpec.show_if_varies`, there's no "always show" mode
    here since a group's whole point is "one column per *varying* key"."""

    field_prefix: str
    getter: Callable[[RowInputs], dict]
    allowed_keys: list[str] | None = None


COLUMN_GROUPS: list[ColumnGroupSpec] = [
    ColumnGroupSpec("hp", _row_hyperparams, HYPERPARAM_DISPLAY_ALLOWLIST),
    ColumnGroupSpec("env", _row_env),
]


def _base_row(
    inputs: RowInputs,
    *,
    comparison_key: str,
    variant: str,
    n_samples: int | None,
    n_features: int | None,
    profile_url: str | None,
    profile_url_label: str | None,
    json_url: str | None,
) -> dict:
    row = {
        "comparison_key": comparison_key,
        "estimator": inputs.case.get("algorithm", {}).get("estimator", "unknown"),
        "dataset": _dataset_name(inputs.case),
        "variant": variant,
        "n_samples": n_samples,
        "n_features": n_features,
        "fit_time": None,
        "fit_speedup": None,
        "predict_time": None,
        "predict_speedup": None,
        "profile_url": profile_url,
        "profile_url_label": profile_url_label,
        "json_url": json_url,
        "status": inputs.status,
    }
    for spec in COLUMNS:
        row[spec.field] = spec.getter(inputs)
    for group in COLUMN_GROUPS:
        row[f"__group_{group.field_prefix}"] = group.getter(inputs)
    return row


def _new_row(
    result: MethodResult,
    variant: str,
    comparison_key: str,
    json_url_fn: Callable[[Path], str | None],
    profile_url_fn: Callable[[Path], str | None],
) -> dict:
    inputs = _method_result_inputs(result)
    return _base_row(
        inputs,
        comparison_key=comparison_key,
        variant=variant,
        n_samples=inputs.data_desc.get("samples"),
        n_features=inputs.data_desc.get("features"),
        profile_url=_profile_url(result, profile_url_fn),
        profile_url_label=_profile_label(result.record),
        json_url=_record_json_url(result, json_url_fn),
    )


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
    inputs = _failed_record_inputs(record)
    return _base_row(
        inputs,
        comparison_key=comparison_key,
        variant=variant,
        n_samples=inputs.data_desc.get("samples"),
        n_features=inputs.data_desc.get("features"),
        profile_url=None,
        profile_url_label=None,
        json_url=json_url_fn(record.record_path) if record.record_path else None,
    )


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


def _varies_across(rows: list[dict], field: str) -> bool:
    return len({_format_value(row.get(field)) for row in rows}) > 1


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

    # Every ColumnGroupSpec (hyperparams, env vars) expands into one
    # Tabulator field per key that varies across rows - resolved once here,
    # then applied uniformly when flattening each row below.
    group_fields: dict[str, dict[str, str]] = {}
    for group in COLUMN_GROUPS:
        group_key = f"__group_{group.field_prefix}"
        raw_dicts = [row[group_key] for row in rows_by_key.values()]
        candidate_keys = (
            group.allowed_keys
            if group.allowed_keys is not None
            else sorted({name for raw in raw_dicts for name in raw})
        )
        varying_names = [
            name
            for name in candidate_keys
            if len({_format_value(raw.get(name)) for raw in raw_dicts}) > 1
        ]
        group_fields[group.field_prefix] = {
            name: f"{group.field_prefix}_{index}"
            for index, name in enumerate(varying_names)
        }

    rows = []
    for row in rows_by_key.values():
        for group in COLUMN_GROUPS:
            raw = row.pop(f"__group_{group.field_prefix}")
            for name, field in group_fields[group.field_prefix].items():
                row[field] = _format_value(raw.get(name))
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
    for spec in COLUMNS:
        if spec.show_if_varies and not _varies_across(rows, spec.field):
            continue
        columns.append(
            _column(spec.title, spec.field, header_filter=True, sorter=spec.sorter)
        )
    for group in COLUMN_GROUPS:
        columns.extend(
            _column(name, field, header_filter=True, sorter="string")
            for name, field in group_fields[group.field_prefix].items()
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
