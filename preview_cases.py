#!/usr/bin/env python3
"""Preview expanded sklbench cases for one PARAMETERS_SETS entry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from sklbench.utils.config import (
    bench_case_filter,
    expand_ranges_in_template,
    expand_template,
    expand_variant_keys,
    merge_dicts,
    parse_cli_parameters,
    remove_duplicated_bench_cases,
    resolve_include_config_path,
)
from sklbench.utils.special_params import (
    assign_case_special_values_on_generation,
    assign_template_special_values,
)


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config_content = json.load(config_file)

    include_content: dict[str, Any] = {}
    for include_config in config_content.get("INCLUDE", []):
        include_config = resolve_include_config_path(include_config)
        include_path = config_path.parent / include_config
        if os.path.isfile(include_path):
            with include_path.open("r", encoding="utf-8") as include_file:
                include_content.update(json.load(include_file)["PARAMETERS_SETS"])
        else:
            raise FileNotFoundError(f"Include file '{include_path}' not found")

    if include_content:
        if "PARAMETERS_SETS" in config_content:
            config_content["PARAMETERS_SETS"].update(include_content)
        else:
            config_content["PARAMETERS_SETS"] = include_content

    if "PARAMETERS_SETS" not in config_content:
        raise ValueError(f"{config_path} does not contain PARAMETERS_SETS")

    for param_set in config_content["PARAMETERS_SETS"].values():
        expand_variant_keys(param_set)

    return config_content


def _parameter_set_templates(
    config_content: dict[str, Any], parameter_set_name: str
) -> list[dict[str, Any]]:
    def normalize_parameter_set(value: Any, name: str) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [deepcopy(value)]
        if isinstance(value, list) and all(
            isinstance(element, dict) for element in value
        ):
            return deepcopy(value)

        raise TypeError(f"Parameter set '{name}' must be a dict or list of dicts")

    parameter_sets = config_content["PARAMETERS_SETS"]
    if parameter_set_name not in parameter_sets:
        available_sets = ", ".join(sorted(parameter_sets))
        raise KeyError(
            f"Unknown parameter set '{parameter_set_name}'. "
            f"Available parameter sets: {available_sets}"
        )

    templates = normalize_parameter_set(
        parameter_sets[parameter_set_name], parameter_set_name
    )

    variants_name = f"{parameter_set_name}+"
    if variants_name in parameter_sets:
        variants = normalize_parameter_set(parameter_sets[variants_name], variants_name)
        templates = [
            merge_dicts(template, variant)
            for template in templates
            for variant in variants
        ]

    return templates


def expand_parameter_set(
    config_path: Path,
    parameter_set_name: str,
    parameters: list[str] | None = None,
    filters: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand one PARAMETERS_SETS entry using sklbench generation semantics."""
    config_content = _load_config(config_path)
    templates = _parameter_set_templates(config_content, parameter_set_name)

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


def _expand_filters(raw_filters: list[str]) -> list[dict[str, Any]]:
    if not raw_filters:
        return []

    filter_template = parse_cli_parameters(raw_filters)
    filter_template = assign_template_special_values(filter_template)
    expand_ranges_in_template(filter_template)
    filter_cases = expand_template(filter_template, [{}], [])
    filter_cases = [
        assign_case_special_values_on_generation(filter_case)
        for filter_case in filter_cases
    ]
    return remove_duplicated_bench_cases(filter_cases)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand one PARAMETERS_SETS entry from a sklbench JSON config "
            "without running benchmarks."
        )
    )
    parser.add_argument("config", type=Path, help="Path to a sklbench JSON config.")
    parser.add_argument(
        "parameter_set",
        nargs="?",
        help=(
            "Name of the PARAMETERS_SETS entry to expand. If omitted with "
            "--list-sets, only available set names are printed."
        ),
    )
    parser.add_argument(
        "--parameters",
        "--params",
        "-p",
        default=[],
        nargs="+",
        help=(
            "CLI-style parameters to merge on top of the selected set, e.g. "
            "'algorithm:library=sklearn data:dtype=float32'."
        ),
    )
    parser.add_argument(
        "--parameter-filters",
        "--filters",
        "-f",
        default=[],
        nargs="+",
        help=(
            "CLI-style filters applied after expansion, e.g. "
            "'algorithm:estimator=RandomForestRegressor'."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level. Use 0 for compact JSON.",
    )
    parser.add_argument(
        "--list-sets",
        action="store_true",
        help="Print available PARAMETERS_SETS names and exit.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Write {'bench_cases': [...]} instead of a bare JSON list.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print only the number of expanded cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        config_content = _load_config(args.config)

        if args.list_sets:
            for parameter_set_name in sorted(config_content["PARAMETERS_SETS"]):
                if parameter_set_name.endswith("+"):
                    continue
                print(parameter_set_name)
            return 0

        if args.parameter_set is None:
            raise ValueError("parameter_set is required unless --list-sets is used")

        cases = expand_parameter_set(
            args.config,
            args.parameter_set,
            parameters=args.parameters,
            filters=args.parameter_filters,
        )
        if args.count:
            output = str(len(cases))
            if args.output is None:
                print(output)
            else:
                args.output.write_text(output + "\n", encoding="utf-8")
            return 0

        payload: Any = {"bench_cases": cases} if args.wrap else cases
        json_kwargs = (
            {"separators": (",", ":")} if args.indent == 0 else {"indent": args.indent}
        )
        output = json.dumps(payload, **json_kwargs)

        if args.output is None:
            print(output)
        else:
            args.output.write_text(output + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
