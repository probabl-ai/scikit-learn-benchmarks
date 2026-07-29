from _common import exclude_estimators
from _implementations import implementations_for_pixi_env, with_implementations
from _linear import linear_array_api_cases


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env(workload="array_api")

    workload = linear_array_api_cases("fast")
    if any(impl["library"] == "sklearnex" for impl in implementations):
        workload = exclude_estimators(workload, {"RidgeClassifier"})

    return with_implementations(workload, implementations)
