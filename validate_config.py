#!/usr/bin/env python3
"""Validate sklbench config expansion, data definitions, and algorithms."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preview_cases import _load_config, _expand_filters, _parameter_set_templates
from sklbench.benchmarks.custom_function import get_function_instance
from sklbench.benchmarks.sklearn_estimator import get_estimator
from sklbench.datasets.loaders import dataset_loading_functions
from sklbench.utils.bench_case import get_bench_case_value
from sklbench.utils.common import flatten_list, hash_from_json_repr
from sklbench.utils.config import (
    bench_case_filter,
    expand_ranges_in_template,
    expand_template,
    expand_variant_keys,
    merge_dicts,
    parse_cli_parameters,
    remove_duplicated_bench_cases,
)
from sklbench.utils.special_params import (
    assign_case_special_values_on_generation,
    assign_template_special_values,
)


@dataclass
class Finding:
    severity: str
    scope: str
    message: str


def _case_label(case_index: int, case_hash: str) -> str:
    return f"case[{case_index}]#{case_hash}"


def _add_error(findings: list[Finding], scope: str, message: str) -> None:
    findings.append(Finding("error", scope, message))


def _add_warning(findings: list[Finding], scope: str, message: str) -> None:
    findings.append(Finding("warning", scope, message))


def _expand_templates(
    templates: list[dict[str, Any]],
    parameters: list[str] | None,
    filters: list[str] | None,
) -> list[dict[str, Any]]:
    global_parameters = parse_cli_parameters(parameters or [])
    if global_parameters:
        templates = [merge_dicts(template, global_parameters) for template in templates]

    templates = [assign_template_special_values(template) for template in templates]
    for template in templates:
        expand_ranges_in_template(template)

    cases: list[dict[str, Any]] = []
    for template in templates:
        cases.extend(expand_template(template, [{}], []))

    cases = [assign_case_special_values_on_generation(case) for case in cases]
    cases = remove_duplicated_bench_cases(cases)

    filter_cases = _expand_filters(filters or [])
    if filter_cases:
        cases = [case for case in cases if bench_case_filter(case, filter_cases)]

    return cases


def _template_templates(
    config_content: dict[str, Any], template_name: str
) -> list[dict[str, Any]]:
    templates_config = config_content.get("TEMPLATES", {})
    if template_name not in templates_config:
        available_templates = ", ".join(sorted(templates_config))
        raise KeyError(
            f"Unknown template '{template_name}'. "
            f"Available templates: {available_templates}"
        )

    template_content = json.loads(json.dumps(templates_config[template_name]))
    expand_variant_keys(template_content)
    templates = [{}]
    for param_set_name in template_content.pop("SETS", []):
        param_set_templates = _parameter_set_templates(config_content, param_set_name)
        templates = flatten_list(
            [
                [
                    merge_dicts(template, param_set_template)
                    for param_set_template in param_set_templates
                ]
                for template in templates
            ]
        )

    return [merge_dicts(template, template_content) for template in templates]


def _load_cases(
    config_content: dict[str, Any],
    target: str,
    target_kind: str,
    parameters: list[str] | None,
    filters: list[str] | None,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    try:
        if target_kind == "template":
            templates = _template_templates(config_content, target)
        else:
            templates = _parameter_set_templates(config_content, target)
        cases = _expand_templates(templates, parameters=parameters, filters=filters)
    except Exception as exc:
        _add_error(findings, target, f"failed to expand {target_kind}: {exc}")
        return [], findings

    if not cases:
        _add_error(findings, target, "expansion produced no cases")
    return cases, findings


def _validate_data(
    case: dict[str, Any],
    scope: str,
    findings: list[Finding],
    materialize_synthetic_data: bool,
) -> str:
    data = get_bench_case_value(case, "data")
    if data is None:
        _add_error(findings, scope, "missing required top-level 'data' section")
        return hash_from_json_repr({})

    data_hash = hash_from_json_repr(data)
    dataset = get_bench_case_value(case, "data:dataset")
    source = get_bench_case_value(case, "data:source")

    if dataset is None and source is None:
        _add_error(findings, scope, "data must define either 'dataset' or 'source'")
        return data_hash
    if dataset is not None and source is not None:
        _add_warning(
            findings,
            scope,
            "data defines both 'dataset' and 'source'; sklbench will use 'dataset'",
        )

    if dataset is not None:
        if dataset not in dataset_loading_functions:
            _add_warning(
                findings,
                scope,
                f"dataset '{dataset}' is not a registered sklbench dataset; "
                "runner will try loading it from cache",
            )
        return data_hash

    if source == "fetch_openml":
        if get_bench_case_value(case, "data:id") is None:
            _add_error(findings, scope, "fetch_openml data source requires 'data:id'")
        return data_hash

    if isinstance(source, str) and source.startswith("make_"):
        from sklearn import datasets as sklearn_datasets
        from sklbench.datasets.synthetic import (
            make_trees_classification_data,
            make_trees_regression_data,
        )

        generation_kwargs = get_bench_case_value(case, "data:generation_kwargs", {})
        if not isinstance(generation_kwargs, dict):
            _add_error(findings, scope, "data:generation_kwargs must be a dict")
            return data_hash

        custom_generators = {
            "make_trees_classification_data": make_trees_classification_data,
            "make_trees_regression_data": make_trees_regression_data,
        }
        generator = custom_generators.get(
            source, getattr(sklearn_datasets, source, None)
        )
        if generator is None:
            _add_error(findings, scope, f"unknown synthetic data generator '{source}'")
            return data_hash

        try:
            inspect.signature(generator).bind_partial(**generation_kwargs)
        except TypeError as exc:
            _add_error(
                findings,
                scope,
                f"invalid generation kwargs for {source}: {exc}",
            )
            return data_hash

        if materialize_synthetic_data:
            try:
                generator(random_state=42, **generation_kwargs)
            except Exception as exc:
                _add_error(
                    findings,
                    scope,
                    f"{source} failed with configured kwargs: {exc}",
                )
        return data_hash

    _add_error(findings, scope, f"unknown data source '{source}'")
    return data_hash


def _validate_estimator_params(
    estimator_class: type, estimator_params: Any, scope: str, findings: list[Finding]
) -> None:
    if estimator_params is None:
        estimator_params = {}
    if not isinstance(estimator_params, dict):
        _add_error(findings, scope, "algorithm:estimator_params must be a dict")
        return

    try:
        signature = inspect.signature(estimator_class.__init__)
    except (TypeError, ValueError) as exc:
        _add_warning(
            findings,
            scope,
            f"could not inspect {estimator_class.__name__} constructor: {exc}",
        )
        return

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    valid_params = set(signature.parameters) - {"self"}
    if not accepts_kwargs:
        unsupported = sorted(set(estimator_params) - valid_params)
        if unsupported:
            _add_error(
                findings,
                scope,
                f"unsupported estimator params for {estimator_class.__name__}: "
                f"{', '.join(unsupported)}",
            )
            return

    try:
        estimator_class(**estimator_params)
    except Exception as exc:
        _add_error(
            findings,
            scope,
            f"could not instantiate {estimator_class.__name__} with params: {exc}",
        )


def _validate_function_kwargs(
    function_instance: Any, kwargs: Any, scope: str, findings: list[Finding]
) -> None:
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        _add_error(findings, scope, "algorithm:kwargs must be a dict")
        return

    try:
        signature = inspect.signature(function_instance)
        signature.bind_partial(**kwargs)
    except TypeError as exc:
        _add_error(findings, scope, f"invalid function kwargs: {exc}")
    except ValueError as exc:
        _add_warning(findings, scope, f"could not inspect function signature: {exc}")


def _validate_algorithm(
    case: dict[str, Any], scope: str, findings: list[Finding]
) -> None:
    allowed_top_level_keys = {"bench", "algorithm", "data", "implementation"}
    unknown_top_level_keys = sorted(set(case) - allowed_top_level_keys)
    if unknown_top_level_keys:
        _add_error(
            findings,
            scope,
            f"unknown top-level keys: {', '.join(unknown_top_level_keys)}; "
            "expected only bench, algorithm, data, and implementation",
        )

    algorithm = get_bench_case_value(case, "algorithm")
    if algorithm is None:
        _add_error(findings, scope, "missing required top-level 'algorithm' section")
        return

    library = get_bench_case_value(case, "implementation:library")
    estimator = get_bench_case_value(case, "algorithm:estimator")
    function = get_bench_case_value(case, "algorithm:function")

    if library is None:
        _add_error(findings, scope, "implementation:library is required")
    if estimator is None and function is None:
        _add_error(
            findings,
            scope,
            "algorithm must define either 'estimator' or 'function'",
        )
        return
    if library is None:
        return
    if estimator is not None and function is not None:
        _add_error(
            findings,
            scope,
            "algorithm must not define both 'estimator' and 'function'",
        )
        return

    if estimator is not None:
        try:
            estimator_class = get_estimator(library, estimator)
        except Exception as exc:
            _add_error(
                findings,
                scope,
                f"could not import estimator '{estimator}' from '{library}': {exc}",
            )
            return
        _validate_estimator_params(
            estimator_class,
            get_bench_case_value(case, "algorithm:estimator_params", {}),
            scope,
            findings,
        )
        return

    try:
        function_instance = get_function_instance(library, function)
    except Exception as exc:
        _add_error(
            findings,
            scope,
            f"could not import function '{function}' from '{library}': {exc}",
        )
        return
    _validate_function_kwargs(
        function_instance,
        get_bench_case_value(case, "algorithm:kwargs", {}),
        scope,
        findings,
    )


def validate_config(
    config_path: Path,
    parameter_sets: list[str] | None,
    parameters: list[str] | None = None,
    filters: list[str] | None = None,
    materialize_synthetic_data: bool = False,
) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    try:
        config_content = _load_config(config_path)
    except Exception as exc:
        return {}, [Finding("error", str(config_path), f"invalid config: {exc}")]

    all_parameter_sets = [
        name
        for name in sorted(config_content["PARAMETERS_SETS"])
        if not name.endswith("+")
    ]
    all_templates = sorted(config_content.get("TEMPLATES", {}))
    default_targets = all_templates or all_parameter_sets
    known_targets = set(all_templates) | set(all_parameter_sets)
    selected_targets = parameter_sets or default_targets
    unknown_targets = sorted(set(selected_targets) - known_targets)
    for unknown_target in unknown_targets:
        _add_error(findings, unknown_target, "unknown template or parameter set")
    selected_targets = [name for name in selected_targets if name in known_targets]

    summary: dict[str, Any] = {
        "config": str(config_path),
        "targets": {},
        "cases": 0,
        "unique_case_hashes": 0,
        "unique_data_hashes": 0,
    }

    all_case_hashes: list[str] = []
    all_data_hashes: list[str] = []
    for target in selected_targets:
        target_kind = "template" if target in all_templates else "parameter_set"
        cases, load_findings = _load_cases(
            config_content,
            target,
            target_kind,
            parameters=parameters,
            filters=filters,
        )
        findings.extend(load_findings)

        case_hashes: list[str] = []
        data_hashes: list[str] = []
        for case_index, case in enumerate(cases):
            case_hash = hash_from_json_repr(case)
            case_hashes.append(case_hash)
            scope = f"{target}:{_case_label(case_index, case_hash)}"
            data_hash = _validate_data(
                case, scope, findings, materialize_synthetic_data
            )
            data_hashes.append(data_hash)
            _validate_algorithm(case, scope, findings)

        _add_duplicate_findings(target, "case", case_hashes, findings)
        _add_duplicate_findings(target, "data", data_hashes, findings)

        summary["targets"][target] = {
            "kind": target_kind,
            "cases": len(cases),
            "unique_case_hashes": len(set(case_hashes)),
            "unique_data_hashes": len(set(data_hashes)),
        }
        all_case_hashes.extend(case_hashes)
        all_data_hashes.extend(data_hashes)

    summary["cases"] = len(all_case_hashes)
    summary["unique_case_hashes"] = len(set(all_case_hashes))
    summary["unique_data_hashes"] = len(set(all_data_hashes))
    return summary, findings


def _add_duplicate_findings(
    parameter_set: str,
    kind: str,
    hashes: list[str],
    findings: list[Finding],
) -> None:
    duplicates = {key: count for key, count in Counter(hashes).items() if count > 1}
    if not duplicates:
        return

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, hash_value in enumerate(hashes):
        if hash_value in duplicates:
            grouped_indices[hash_value].append(index)

    for hash_value, indices in sorted(grouped_indices.items()):
        _add_warning(
            findings,
            parameter_set,
            f"duplicate {kind} hash {hash_value} occurs in cases {indices}",
        )


def _print_text_report(
    summary: dict[str, Any], findings: list[Finding], max_findings: int
) -> None:
    print(f"config: {summary.get('config')}")
    print(f"cases: {summary.get('cases', 0)}")
    print(f"unique case hashes: {summary.get('unique_case_hashes', 0)}")
    print(f"unique data hashes: {summary.get('unique_data_hashes', 0)}")

    for target, set_summary in summary.get("targets", {}).items():
        print(
            f"- {target} ({set_summary['kind']}): {set_summary['cases']} cases, "
            f"{set_summary['unique_data_hashes']} unique data hashes"
        )

    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")

    displayed_findings = findings if max_findings == 0 else findings[:max_findings]
    for finding in displayed_findings:
        print(f"{finding.severity}: {finding.scope}: {finding.message}")
    if max_findings and len(findings) > max_findings:
        print(
            f"... {len(findings) - max_findings} more findings hidden; "
            "use --json for the full report or --max-findings 0 for full text."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate expanded sklbench config cases without running benchmarks."
    )
    parser.add_argument("config", type=Path, help="Path to a sklbench JSON config.")
    parser.add_argument(
        "parameter_sets",
        nargs="*",
        help=(
            "Specific TEMPLATES or PARAMETERS_SETS names to validate. Defaults to "
            "all templates when present, otherwise all non-variant parameter sets."
        ),
    )
    parser.add_argument(
        "--parameters",
        "--params",
        "-p",
        default=[],
        nargs="+",
        help=(
            "CLI-style parameters to merge on top of each selected set before "
            "validation."
        ),
    )
    parser.add_argument(
        "--parameter-filters",
        "--filters",
        "-f",
        default=[],
        nargs="+",
        help="CLI-style filters applied after expansion and before validation.",
    )
    parser.add_argument(
        "-m",
        "--materialize-synthetic-data",
        action="store_true",
        help=(
            "Call synthetic data generators with configured kwargs. "
            "This catches value errors but can allocate large arrays."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON validation report.",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=50,
        help="Maximum findings to print in text mode. Use 0 for all findings.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary, findings = validate_config(
        args.config,
        args.parameter_sets or None,
        parameters=args.parameters,
        filters=args.parameter_filters,
        materialize_synthetic_data=args.materialize_synthetic_data,
    )

    payload = {
        "summary": summary,
        "findings": [finding.__dict__ for finding in findings],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_text_report(summary, findings, args.max_findings)

    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
