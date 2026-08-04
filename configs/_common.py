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


def disable_profiling_for_array_api_gpu_cases(cases: Iterable[dict]) -> None:
    """py-spy is consistently unreliable on array-API + GPU cases on this
    (very low-tier) GPU: order="F" input forces an expensive contiguous-copy
    / fresh GPU memory allocation before solvers like dpnp's SVD can run
    (same root cause as
    https://github.com/uxlfoundation/scikit-learn-intelex/issues/3235, but
    hit here via dpnp's own array-API path rather than oneDAL), and that
    allocation is already borderline slow untraced - attaching py-spy at all
    (even without --native) is enough to make it fall pathologically behind.
    Rather than special-casing order="F", disable profiling for every
    array-API + GPU case, since py-spy errors keep showing up there more
    broadly on this hardware.

    Replaces `bench` on matching cases with a new dict rather than mutating
    it in place - config generators like `real_datasets.py` reuse the same
    `bench` dict object across many cases, so an in-place mutation here would
    leak onto unrelated cases sharing that object.
    """
    for case in cases:
        impl = case["implementation"]
        context = {**(impl.get("sklearn_context") or {}), **(impl.get("sklearnex_context") or {})}
        if impl.get("device") == "gpu" and context.get("array_api_dispatch", False):
            case["bench"] = {**case.get("bench", {}), "py_spy_profiling": False}
