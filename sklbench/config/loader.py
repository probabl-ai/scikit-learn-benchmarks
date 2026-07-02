from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import ValidationError

from .models import BaseCase, EstimatorCase, PipelineCase


Case = EstimatorCase | PipelineCase


def _json_normalize(value: Any, context: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be JSON serializable: {exc}") from exc


def validate_case(case: dict | BaseCase) -> Case:
    if isinstance(case, (EstimatorCase, PipelineCase)):
        _json_normalize(case.json_dict(), "case")
        return case
    if isinstance(case, BaseCase):
        raise TypeError(f"unsupported case model: {type(case).__name__}")
    if not isinstance(case, dict):
        raise TypeError(f"case must be a dict or BaseCase, got {type(case).__name__}")

    normalized_input = _json_normalize(case, "case")
    try:
        if "algorithm" in normalized_input or "implementation" in normalized_input:
            return EstimatorCase.model_validate(normalized_input)
        return PipelineCase.model_validate(normalized_input)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _load_module_from_path(path: Path) -> ModuleType:
    module_name = "_sklbench_config"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import config script: {path}")

    module = importlib.util.module_from_spec(spec)
    script_dir = str(path.parent.resolve())
    inserted = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(script_dir)
    return module


def load_cases_from_script(path: str | Path) -> list[Case]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config script not found: {config_path}")
    if config_path.suffix != ".py":
        raise ValueError(f"Config must be a Python script: {config_path}")

    module = _load_module_from_path(config_path)
    generate_cases = getattr(module, "generate_cases", None)
    if generate_cases is None:
        raise ValueError(f"{config_path} does not define generate_cases()")
    if not callable(generate_cases):
        raise TypeError(f"{config_path}:generate_cases is not callable")

    raw_cases = generate_cases()
    if not isinstance(raw_cases, list):
        raise TypeError(
            f"{config_path}:generate_cases() must return a list, "
            f"got {type(raw_cases).__name__}"
        )

    cases = []
    for index, case in enumerate(raw_cases):
        try:
            cases.append(validate_case(case))
        except Exception as exc:
            raise ValueError(f"Invalid case at index {index}: {exc}") from exc
    return cases
