import configs.all_models as all_models


def generate_cases() -> list[dict]:
    return [
        case
        for case in all_models.generate_cases()
        if case.algorithm.estimator == "KMeans"
    ]
