"""The KMeans cases of `all_models.py`.

The time limit is raised from `all_models.py`'s 300s: on one socket of the
Xeon server, scikit-learn `main` takes more than 60s per fit on sift with
k=100, so its 5 runs did not fit in 300s.
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
