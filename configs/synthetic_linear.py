from itertools import product
from math import ceil
from typing import Iterable

from sklbench.config import Implementation


def _is_array_api(implem: dict) -> bool:
    return Implementation(**implem).is_array_api()


def _synthetic_linear_cases(implem: dict, tier: str) -> Iterable[dict]:
    array_api = _is_array_api(implem)
    bench = {"n_runs": 5, "time_limit": 10} if tier == "fast" else {"n_runs": 3}
    if tier == "fast":
        data_variants = [
            {"n_samples": 5000000, "n_features": 2, "n_informative": 2},
            {"n_samples": 500000, "n_features": 20, "n_informative": 5},
            {"n_samples": 50000, "n_features": 200, "n_informative": 40},
            {"n_samples": 5000, "n_features": 2000, "n_informative": 100},
        ]
    else:
        n_samples = 1000 if array_api else 3000
        data_variants = [
            {
                "n_samples": n_samples,
                "n_features": 20,
                "n_informative": ceil(0.5 * 20),
            }
        ]

    if array_api:
        algorithm_variants = [
            {"estimator": "Ridge", "estimator_params": {"solver": "svd"}},
            {"estimator": "LogisticRegression"},
            {"estimator": "RidgeClassifier", "estimator_params": {"solver": "svd"}},
        ]
    else:
        algorithm_variants = [
            {"estimator": "Ridge"},
            {"estimator": "LinearRegression"},
            {"estimator": "LogisticRegression", "estimator_params": {"solver": "lbfgs"}},
            {
                "estimator": "LogisticRegression",
                "estimator_params": {"solver": "newton-cg"},
            },
        ]

    for algorithm, generation_kwargs in product(algorithm_variants, data_variants):
        is_classifier = algorithm["estimator"] in {
            "LogisticRegression",
            "RidgeClassifier",
        }
        source = "make_classification" if is_classifier else "make_regression"
        extra_generation_kwargs = (
            {"n_classes": 2, "n_redundant": 0} if is_classifier else {}
        )
        data = {
            "source": source,
            "generation_kwargs": {**generation_kwargs, **extra_generation_kwargs},
        }
        if not array_api or algorithm["estimator"] == "Ridge":
            data["order"] = "C"
        yield {
            "bench": bench,
            "algorithm": algorithm,
            "data": data,
            "implementation": implem,
        }


def generate_cases(implem: dict | None = None, tier: str = "normal") -> list[dict]:
    if implem is None:
        implem = {"library": "sklearn"}
    return list(_synthetic_linear_cases(implem, tier))
