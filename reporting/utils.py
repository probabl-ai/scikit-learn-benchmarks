import json
from typing import Any, Dict, List, Tuple
from copy import deepcopy
import itertools
from collections.abc import Iterable, Callable


def partition_iterable[T](
    iterable: Iterable[T],
    predicate: Callable[[T], bool],
) -> tuple[list[T], list[T]]:
    yes: list[T] = []
    no: list[T] = []

    for x in iterable:
        (yes if predicate(x) else no).append(x)

    return yes, no


def groupby[T, K](
    iterable: Iterable[T],
    keyfunc: Callable[[T], K],
) -> dict[K, list[T]]:
    sorted_iterable = sorted(iterable, key=keyfunc)
    return {
        k: list(g)
        for k, g in itertools.groupby(sorted_iterable, keyfunc)
    }


def without_keys(value: Any, excluded_names: set, excluded_prefixes: tuple = ()) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, nested_value in value.items():
            if key in excluded_names or any(
                key.startswith(prefix) for prefix in excluded_prefixes
            ):
                continue
            result[key] = without_keys(nested_value, excluded_names, excluded_prefixes)
        return result
    if isinstance(value, list):
        return [without_keys(item, excluded_names, excluded_prefixes) for item in value]
    return deepcopy(value)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
