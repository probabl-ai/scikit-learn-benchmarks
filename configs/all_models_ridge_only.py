from all_models import generate_cases as generate_all_models_cases


def generate_cases() -> list[dict]:
    # all_models.generate_cases() returns EstimatorCase objects, not plain
    # dicts, despite its own type hint - its last filtering step
    # (filter_array_api_supported_cases_if_needed) converts every case via
    # EstimatorCase(**case) before yielding it.
    return [
        case
        for case in generate_all_models_cases()
        if case.algorithm.estimator == "Ridge"
    ]
