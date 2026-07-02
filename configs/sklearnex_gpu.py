from _generators import (
    linear_array_api_cases,
    linear_cases,
    sklearnex_gpu_implementation,
    with_implementations,
)


def generate_cases() -> list[dict]:
    implementation = sklearnex_gpu_implementation()
    cases = []
    cases.extend(with_implementations(linear_cases(), implementation))
    cases.extend(with_implementations(linear_array_api_cases(), implementation))
    return cases
