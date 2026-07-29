from itertools import product
from math import ceil
from typing import Iterable


def linear_cases(template: str) -> Iterable[dict]:
    return _linear_cases(template, array_api=False)


def linear_array_api_cases(template: str) -> Iterable[dict]:
    return _linear_cases(template, array_api=True)


def _linear_cases(template: str, *, array_api: bool) -> Iterable[dict]:
    bench = {"n_runs": 5, "time_limit": 10} if template == "fast" else {"n_runs": 3}
    if template == "fast":
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
        }

    if template != "fast" and not array_api:
        yield from _real_data_linear_cases(bench)


def _real_data_linear_cases(bench: dict) -> Iterable[dict]:
    """Real-dataset cases exercising the `linear` preprocessing kind.

    `ames_housing` has both categorical (mostly `object`-dtype) and numeric
    columns with real missing values, unlike the synthetic data above.
    """
    yield {
        "bench": bench,
        "algorithm": {"estimator": "Ridge"},
        "data": {"dataset": "ames_housing", "preprocessing_kind": "linear"},
    }
