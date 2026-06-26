from _common import (
    apply_sklearnex_cpu_tree_variants,
    clustering_cases,
    exclude_estimators,
    linear_array_api_cases,
    linear_cases,
    sklearnex_cpu_implementation,
    tree_cases,
    with_implementations,
)


def generate_cases() -> list[dict]:
    implementation = sklearnex_cpu_implementation()
    cases = []
    cases.extend(
        with_implementations(
            apply_sklearnex_cpu_tree_variants(tree_cases()), implementation
        )
    )
    cases.extend(with_implementations(linear_cases(), implementation))
    cases.extend(
        with_implementations(
            exclude_estimators(linear_array_api_cases(), {"RidgeClassifier"}),
            implementation,
        )
    )
    cases.extend(with_implementations(clustering_cases(), implementation))
    return cases
