"""Small matrix for native cuML vs the wrappers' sklearn-like APIs.

Run on a CUDA machine with the ``cuml`` Pixi environment:

    pixi run -e cuml python -m sklbench --config configs/cuml_smoke.py \\
        --results-dir ./results/tests/
"""

from sklbench.config.utils import filter_gpu_cases_if_unavailable

from _utils.implementations import implementations_for_pixi_env


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env()

    cases = []
    for implem in implementations:
        cases.extend(
            [
                {
                    "algorithm": {
                        "estimator": "LogisticRegression",
                        "estimator_params": {"solver": "lbfgs"},
                    },
                    "data": {
                        "source": "make_classification",
                        "generation_kwargs": {
                            "n_samples": 2000,
                            "n_features": 20,
                            "n_informative": 10,
                            "n_classes": 2,
                            "random_state": 0,
                        },
                    },
                    "implementation": implem,
                },
                {
                    "algorithm": {
                        "estimator": "Ridge",
                        "estimator_params": {"alpha": 1.0},
                    },
                    "data": {
                        "source": "make_regression",
                        "generation_kwargs": {
                            "n_samples": 2000,
                            "n_features": 20,
                            "n_informative": 10,
                            "random_state": 0,
                        },
                    },
                    "implementation": implem,
                },
                {
                    "algorithm": {
                        "estimator": "RandomForestClassifier",
                        "estimator_params": {
                            "n_estimators": 32,
                            "max_depth": 8,
                            "max_features": 0.3,
                        },
                    },
                    "data": {
                        "source": "make_classification",
                        "generation_kwargs": {
                            "n_samples": 2000,
                            "n_features": 20,
                            "n_informative": 10,
                            "n_classes": 2,
                            "random_state": 0,
                        },
                    },
                    "implementation": implem,
                },
                {
                    "algorithm": {
                        "estimator": "RandomForestRegressor",
                        "estimator_params": {
                            "n_estimators": 32,
                            "max_depth": 8,
                            "max_features": 0.3,
                        },
                    },
                    "data": {
                        "source": "make_regression",
                        "generation_kwargs": {
                            "n_samples": 2000,
                            "n_features": 20,
                            "n_informative": 10,
                            "random_state": 0,
                        },
                    },
                    "implementation": implem,
                },
                {
                    "algorithm": {
                        "estimator": "KMeans",
                        "estimator_params": {"n_clusters": 5, "n_init": 1},
                    },
                    "data": {
                        "source": "make_blobs",
                        "generation_kwargs": {
                            "centers": 5,
                            "n_samples": 2000,
                            "n_features": 10,
                            "random_state": 0,
                        },
                    },
                    "implementation": implem,
                },
            ]
        )

    for case in cases:
        case.setdefault("bench", {})
        case["bench"].setdefault("n_runs", 1)
        case["bench"].setdefault("time_limit", 60)

    return list(filter_gpu_cases_if_unavailable(cases))
