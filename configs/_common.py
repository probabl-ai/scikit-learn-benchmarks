from copy import deepcopy
from typing import Iterable


def _merge_dicts(first: dict, second: dict) -> dict:
    result = deepcopy(first)
    for key, value in second.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def exclude_estimators(cases: Iterable[dict], estimators: set[str]) -> list[dict]:
    return [
        case
        for case in cases
        if case["algorithm"].get("estimator") not in estimators
    ]
