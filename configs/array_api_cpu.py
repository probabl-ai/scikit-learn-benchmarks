from _generators import (
    array_api_cpu_implementations,
    linear_array_api_cases,
    with_implementations,
)


def generate_cases() -> list[dict]:
    return with_implementations(
        linear_array_api_cases(), array_api_cpu_implementations()
    )
