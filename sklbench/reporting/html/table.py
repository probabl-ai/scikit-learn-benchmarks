from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from enum import Enum
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
    one shape, so every column getter below is written once and works for
    both (see `_method_result_inputs` / `_failed_record_inputs`)."""

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


# --- Column value getters -------------------------------------------------
# Each of these computes one column's raw value from RowInputs alone. Add
# one here (plus a ColumnSpec entry below) for any new case/software-derived
# inspection column - see the COLUMNS list's docstring.


def _row_estimator(inputs: RowInputs) -> str:
    return inputs.case.get("algorithm", {}).get("estimator", "unknown")


def _row_dataset(inputs: RowInputs) -> str:
    return _dataset_name(inputs.case)


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


def _row_status(inputs: RowInputs) -> str:
    return inputs.status


class ColumnVisibility(Enum):
    """When a `ColumnSpec` is included in the table at all:
    - ALWAYS: always included.
    - IF_VARIES: only once at least two distinct (formatted) values appear
      across all rows - a column that's constant for the whole table isn't
      worth its width (e.g. "order" when every case happens to use the same
      layout).
    - IF_ANY: only once at least one row's value is truthy - for a field
      that's either meaningfully set or essentially absent, where "varies"
      wouldn't work because every *present* value is expected to differ
      from every other one anyway (e.g. a profile link: each is a distinct
      per-record URL, so a table where every row has one would "vary" even
      though the real question is just "was profiling ever captured?")."""

    ALWAYS = "always"
    IF_VARIES = "if_varies"
    IF_ANY = "if_any"


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the detailed-results table, listed below in `COLUMNS`
    in the exact order they appear on screen - read that list top to bottom
    to see every column the table can show and when.

    `getter`, when set, computes the column's value once per row from
    `RowInputs` alone - the easy path for a new case/software-derived
    inspection column (add a getter above, then one `ColumnSpec` entry to
    `COLUMNS`; nothing else needs to change). Leave it `None` for a column
    whose value some other code already writes into the row dict (fit/predict
    times and speedups from the method loop, profile/JSON links needing the
    caller's URL-building functions - see `_base_row`/`_add_result_method`);
    the entry then serves purely as that column's display/visibility
    declaration.

    `custom_show`, when set, overrides `visibility` with an arbitrary
    `rows -> bool` predicate, for a column none of the three visibility
    modes describes correctly (see `Status` in `COLUMNS`: a table where
    every single row failed shouldn't hide the column just because "failed"
    doesn't *vary* - that's exactly the case it needs to report)."""

    title: str
    field: str
    getter: Callable[[RowInputs], Any] | None = None
    visibility: ColumnVisibility = ColumnVisibility.ALWAYS
    custom_show: Callable[[list[dict]], bool] | None = None
    header_filter: bool = True
    header_sort: bool | None = None
    sorter: str = "string"
    formatter_name: str | None = None
    link_label: str | None = None
    visible: bool | None = None


def _status_column_visible(rows: list[dict]) -> bool:
    return any(row["status"] != "ok" for row in rows)


@dataclass(frozen=True)
class ColumnGroupSpec:
    """One family of dynamically-named columns, one per key that varies
    across rows (e.g. one column per hyperparameter, one per env var) -
    mixed directly into `COLUMNS` at the position its columns should appear.
    `getter` returns this row's whole `{key: raw_value}` dict; `allowed_keys`
    restricts which keys are even considered (None = every key seen across
    all rows). Every generated column is shown only where it varies - a
    group's whole point is "one column per *varying* key", so there's no
    "always show" mode here."""

    field_prefix: str
    getter: Callable[[RowInputs], dict]
    allowed_keys: list[str] | None = None


COLUMNS: list[ColumnSpec | ColumnGroupSpec] = [
    ColumnSpec(
        "comparison_key",
        "comparison_key",
        visible=False,
        header_filter=False,
        header_sort=False,
    ),
    # Title is overridden per-call by `variant_column_title`.
    ColumnSpec("Variant name", "variant"),
    ColumnSpec("Estimator name", "estimator", _row_estimator),
    ColumnSpec("Dataset name", "dataset", _row_dataset),
    # n_samples/n_features: initial value below comes from data_desc via the
    # getter, then gets overridden in `_add_result_method` once the "fit"
    # method's result arrives, since that's the more authoritative source.
    ColumnSpec(
        "n_samples",
        "n_samples",
        lambda inputs: inputs.data_desc.get("samples"),
        sorter="number",
    ),
    ColumnSpec(
        "n_features",
        "n_features",
        lambda inputs: inputs.data_desc.get("features"),
        sorter="number",
    ),
    ColumnSpec("columns", "columns", _row_columns_kind, ColumnVisibility.IF_VARIES),
    ColumnSpec("order", "order", _row_order, ColumnVisibility.IF_VARIES),
    ColumnSpec("max_bins", "max_bins", _row_max_bins, ColumnVisibility.IF_VARIES),
    ColumnSpec("OpenMP", "openmp", _row_openmp, ColumnVisibility.IF_VARIES),
    ColumnSpec(
        "GOMP_SPINCOUNT",
        "gomp_spincount",
        _row_gomp_spincount,
        ColumnVisibility.IF_VARIES,
    ),
    ColumnSpec(
        "KMP_BLOCKTIME",
        "kmp_blocktime",
        _row_kmp_blocktime,
        ColumnVisibility.IF_VARIES,
    ),
    ColumnGroupSpec("hp", _row_hyperparams, HYPERPARAM_DISPLAY_ALLOWLIST),
    ColumnGroupSpec("env", _row_env),
    ColumnSpec(
        "Status", "status", _row_status, custom_show=_status_column_visible
    ),
    ColumnSpec(
        "fit time",
        "fit_time",
        header_filter=False,
        sorter="number",
        formatter_name="duration",
    ),
    ColumnSpec(
        "fit speed up",
        "fit_speedup",
        header_filter=False,
        sorter="number",
        formatter_name="speedup",
    ),
    ColumnSpec(
        "predict time",
        "predict_time",
        header_filter=False,
        sorter="number",
        formatter_name="duration",
    ),
    ColumnSpec(
        "predict speed up",
        "predict_speedup",
        header_filter=False,
        sorter="number",
        formatter_name="speedup",
    ),
    ColumnSpec(
        "profile link",
        "profile_url",
        visibility=ColumnVisibility.IF_ANY,
        header_filter=False,
        header_sort=False,
        formatter_name="link",
        link_label="profile",
    ),
    ColumnSpec(
        "JSON link",
        "json_url",
        header_filter=False,
        header_sort=False,
        formatter_name="link",
        link_label="JSON",
    ),
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
    }
    for spec in COLUMNS:
        if isinstance(spec, ColumnGroupSpec):
            row[f"__group_{spec.field_prefix}"] = spec.getter(inputs)
        elif spec.getter is not None:
            row[spec.field] = spec.getter(inputs)
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


def _column_dict(
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


def _spec_column_dict(spec: ColumnSpec, *, title: str) -> dict:
    return _column_dict(
        title,
        spec.field,
        visible=spec.visible,
        header_filter=spec.header_filter,
        header_sort=spec.header_sort,
        sorter=spec.sorter,
        formatter_name=spec.formatter_name,
        link_label=spec.link_label,
    )


def _spec_visible(spec: ColumnSpec, rows: list[dict]) -> bool:
    if spec.custom_show is not None:
        return spec.custom_show(rows)
    if spec.visibility is ColumnVisibility.ALWAYS:
        return True
    if spec.visibility is ColumnVisibility.IF_VARIES:
        return _varies_across(rows, spec.field)
    if spec.visibility is ColumnVisibility.IF_ANY:
        return any(row.get(spec.field) for row in rows)
    raise AssertionError(spec.visibility)


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
    column_groups = [spec for spec in COLUMNS if isinstance(spec, ColumnGroupSpec)]
    group_fields: dict[str, dict[str, str]] = {}
    for group in column_groups:
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
        for group in column_groups:
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

    columns = []
    for spec in COLUMNS:
        if isinstance(spec, ColumnGroupSpec):
            columns.extend(
                _column_dict(name, field, header_filter=True, sorter="string")
                for name, field in group_fields[spec.field_prefix].items()
            )
            continue
        if not _spec_visible(spec, rows):
            continue
        title = variant_column_title if spec.field == "variant" else spec.title
        columns.append(_spec_column_dict(spec, title=title))

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
