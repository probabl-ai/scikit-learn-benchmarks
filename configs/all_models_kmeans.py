"""The KMeans cases of `all_models.py`.

The time limit is raised from `all_models.py`'s 300s: with threadpoolctl
3.7.0 (see joblib/threadpoolctl#283), the 5 fits of scikit-learn `main` on
sift with k=100 took more than 300s.
"""
from all_models import generate_cases as generate_all_models_cases


def generate_cases() -> list[dict]:
    cases = [
        case for case in generate_all_models_cases()
        if case.algorithm.estimator == "KMeans"
    ]
    for case in cases:
        case.bench.time_limit = 1800
    return cases
