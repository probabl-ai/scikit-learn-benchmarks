from copy import deepcopy
from itertools import product
from typing import Iterable

from joblib import cpu_count

from _common import deterministic_random_choice


def tree_cases(template: str) -> Iterable[dict]:
    return _synthetic_tree_cases() if template == "fast" else _test_tree_cases()


def apply_sklearnex_cpu_tree_variants(cases: Iterable[dict]) -> Iterable[dict]:
    for case in cases:
        yield case
        n_samples = case["data"].get("generation_kwargs", {}).get("n_samples")
        if n_samples is None:
            continue
        variant = deepcopy(case)
        variant["algorithm"]["estimator_params"]["max_bins"] = n_samples
        yield variant


def _synthetic_tree_cases(scale: int = 10) -> Iterable[dict]:
    bench = {"n_runs": 5, "time_limit": scale}

    estimator_params_base = {
        "n_estimators": 64,
        "max_features": 0.3,
        "n_jobs": cpu_count(only_physical_cores=True)
    }
    estimator_param_variants = [{}, {"max_depth": 4}, {"max_leaf_nodes": 1000}]

    base_data = [
        {"n_samples": 10000 * scale, "n_features": 1, "n_informative": 1},
        {"n_samples": 1000 * scale, "n_features": 20, "n_informative": 10},
        {"n_samples": 100 * scale, "n_features": 500, "n_informative": 50},
    ]

    classification_data = []
    for data_spec in base_data:
        data_spec = deepcopy(data_spec)
        data_spec.update({
            "n_classes": 2,
            "n_redundant": 0,
            "columns": "mix"
        })
        if data_spec["n_features"] == 1:
            data_spec["n_clusters_per_class"] = 1
        classification_data.append(data_spec)
        data_spec = deepcopy(data_spec)
        data_spec["columns"] = deterministic_random_choice(
            seed=data_spec,
            choices=["continuous", "binary", "long-tail"]
        )
        classification_data.append(data_spec)

    for estimator, params, generation_kwargs in product(
        ["RandomForestClassifier", "ExtraTreesClassifier"],
        estimator_param_variants,
        classification_data,
    ):
        yield {
            "bench": bench,
            "algorithm": {
                "estimator": estimator,
                "estimator_params": {
                    **estimator_params_base,
                    **params,
                },
            },
            "data": {
                "source": "make_trees_classification_data",
                "generation_kwargs": generation_kwargs,
            },
        }

    regression_data = []
    for data_spec in base_data:
        data_spec = deepcopy(data_spec)
        data_spec.update({
            "noise": 0.1,
            "columns": "mix"
        })
        regression_data.append(data_spec)
        data_spec = deepcopy(data_spec)
        data_spec["columns"] = deterministic_random_choice(
            seed=data_spec,
            choices=["continuous", "binary", "long-tail"]
        )
        regression_data.append(data_spec)

    for estimator, params, generation_kwargs in product(
        ["RandomForestRegressor", "ExtraTreesRegressor"],
        estimator_param_variants,
        regression_data,
    ):
        yield {
            "bench": bench,
            "algorithm": {
                "estimator": estimator,
                "estimator_params": {
                    **estimator_params_base,
                    **params,
                },
            },
            "data": {
                "source": "make_trees_regression_data",
                "generation_kwargs": generation_kwargs,
            },
        }


def _test_tree_cases() -> Iterable[dict]:
    cases_per_model = {}
    for case in _synthetic_tree_cases(scale=2):
        cases_per_model.setdefault(case["algorithm"]["estimator"], []).append(case)

    # select one per algorithm
    for estimator, cases in cases_per_model.items():
        case = deterministic_random_choice(seed=estimator, choices=cases)
        case = deepcopy(case)
        case["bench"]["n_runs"] = 2
        yield case

    yield from _real_data_tree_cases()


def _real_data_tree_cases() -> Iterable[dict]:
    """Real-dataset cases exercising the `trees` preprocessing kind.

    `ames_housing` has both categorical (mostly `object`-dtype) and numeric
    columns with real missing values, unlike the synthetic data above.
    """
    estimator_params = {
        "n_estimators": 64,
        "max_features": 0.3,
        "n_jobs": cpu_count(only_physical_cores=True),
    }
    for estimator in ["RandomForestRegressor", "ExtraTreesRegressor"]:
        yield {
            "bench": {"n_runs": 2},
            "algorithm": {
                "estimator": estimator,
                "estimator_params": estimator_params,
            },
            "data": {"dataset": "ames_housing", "preprocessing_kind": "trees"},
        }
