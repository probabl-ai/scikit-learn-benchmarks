import hashlib
import json
from copy import deepcopy
from typing import Iterable
import random


def deterministic_random_choice(seed: object, choices: list, n: int = 1):
    """Deterministically pick one of `choices` based on the JSON content of `seed`.

    The same `seed` always yields the same choice, regardless of call order or
    process-level random state, so config scripts stay reproducible while still
    varying a parameter across cases without spelling out every combination.
    """
    digest = hashlib.sha256(
        json.dumps(seed, sort_keys=True).encode("utf-8")
    ).digest()
    rng = random.Random(digest)
    if n == 1:
        return rng.choice(choices)
    else:
        return rng.choices(choices, k=n)


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
