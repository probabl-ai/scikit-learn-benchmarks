from _common import (
    clustering_cases,
    linear_array_api_cases,
    linear_cases,
    sklearn_implementation,
    tree_cases,
    with_implementations,
)


def generate_cases() -> list[dict]:
    implementation = sklearn_implementation()
    cases = []
    cases.extend(with_implementations(tree_cases(), implementation))
    cases.extend(with_implementations(linear_cases(), implementation))
    cases.extend(with_implementations(linear_array_api_cases(), implementation))
    cases.extend(with_implementations(clustering_cases(), implementation))
    return cases
