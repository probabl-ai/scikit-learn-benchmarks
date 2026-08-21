"""Two tiny HistGradientBoosting cases, for exercising the PR-comparison
benchmark flow (.github/workflows/pr-comparison.yml) end to end quickly -
small enough to fit+predict in well under a second per run, not meant to
produce representative benchmark numbers.

Exercised live in https://github.com/probabl-ai/scikit-learn-benchmarks/pull/29
and its end-to-end follow-up test PR.
"""


def generate_cases(implem: dict | None = None) -> list[dict]:
    if implem is None:
        implem = {"library": "sklearn"}

    estimator_params = {
        "max_iter": 20,
        "max_leaf_nodes": 15,
        "max_bins": 255,
        "early_stopping": False,
    }
    bench = {"n_runs": 3, "py_spy_profiling": False}

    return [
        {
            "bench": bench,
            "algorithm": {
                "estimator": "HistGradientBoostingClassifier",
                "estimator_params": estimator_params,
            },
            "data": {
                "source": "make_trees_classification_data",
                "generation_kwargs": {
                    "n_samples": 2000,
                    "n_features": 10,
                    "n_informative": 5,
                    "n_classes": 2,
                    "n_redundant": 0,
                    "random_state": 0,
                    "columns": "continuous",
                },
                "order": "C",
            },
            "implementation": implem,
        },
        {
            "bench": bench,
            "algorithm": {
                "estimator": "HistGradientBoostingRegressor",
                "estimator_params": estimator_params,
            },
            "data": {
                "source": "make_trees_regression_data",
                "generation_kwargs": {
                    "n_samples": 2000,
                    "n_features": 10,
                    "n_informative": 5,
                    "random_state": 0,
                    "columns": "continuous",
                },
                "order": "C",
            },
            "implementation": implem,
        },
    ]
