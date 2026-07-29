from copy import deepcopy
from itertools import product
from math import ceil
from typing import Iterable

from joblib import cpu_count


def tree_cases(template: str) -> Iterable[dict]:
    return _fast_tree_cases() if template == "fast" else _test_tree_cases()


def apply_sklearnex_cpu_tree_variants(cases: Iterable[dict]) -> Iterable[dict]:
    for case in cases:
        yield case
        n_samples = case["data"].get("generation_kwargs", {}).get("n_samples")
        if n_samples is None:
            continue
        variant = deepcopy(case)
        variant["algorithm"]["estimator_params"]["max_bins"] = n_samples
        yield variant


def _test_tree_cases() -> Iterable[dict]:
    params = {
        "n_estimators": 16,
        "max_features": 0.5,
        "max_depth": 4,
        "n_jobs": 1,
    }

    base_generation = {
        "n_samples": 1000,
        "n_features": 10,
        "n_informative": 5,
        "columns": "mix",
    }
    clf_data = {
        "source": "make_trees_classification_data",
        "generation_kwargs": {
            **base_generation,
            "n_classes": 2,
        },
    }

    for estimator in ["RandomForestClassifier", "ExtraTreesClassifier"]:
        yield {
            "bench": {"n_runs": 3},
            "algorithm": {
                "estimator": estimator,
                "estimator_params": params,
            },
            "data": clf_data,
        }

    reg_data = {
        "source": "make_trees_regression_data",
        "generation_kwargs": {
            **base_generation,
            "noise": 0.1,
        },
    }

    for estimator in ["RandomForestRegressor", "ExtraTreesRegressor"]:
        yield {
            "bench": {"n_runs": 3},
            "algorithm": {
                "estimator": estimator,
                "estimator_params": params,
            },
            "data": reg_data,
        }


def _fast_tree_cases() -> Iterable[dict]:
    bench = {"n_runs": 5, "time_limit": 10}
    n_jobs = cpu_count(only_physical_cores=True)
    estimator_param_variants = [{}, {"max_depth": 4}, {"max_leaf_nodes": 1000}]

    classification_data = [
        {
            "n_samples": 100000,
            "n_features": 1,
            "n_clusters_per_class": 1,
            "columns": ["continuous", "long-tail"],
        },
        {"n_samples": 10000, "n_features": 20, "columns": ["mix"]},
        {"n_samples": 1000, "n_features": 500, "columns": ["mix", "binary"]},
    ]
    for estimator, params, data_spec in product(
        ["RandomForestClassifier", "ExtraTreesClassifier"],
        estimator_param_variants,
        classification_data,
    ):
        for columns in data_spec["columns"]:
            generation_kwargs = {
                key: value for key, value in data_spec.items() if key != "columns"
            }
            generation_kwargs.update(
                {
                    "columns": columns,
                    "n_classes": 2,
                    "n_redundant": 0,
                    "n_informative": ceil(0.5 * data_spec["n_features"]),
                }
            )
            yield {
                "bench": bench,
                "algorithm": {
                    "estimator": estimator,
                    "estimator_params": {
                        "n_estimators": 64,
                        "max_features": 0.3,
                        "n_jobs": n_jobs,
                        **params,
                    },
                },
                "data": {
                    "source": "make_trees_classification_data",
                    "generation_kwargs": generation_kwargs,
                },
            }

    regression_data = [
        {"n_samples": 100000, "n_features": 1, "columns": ["continuous", "long-tail"]},
        {"n_samples": 10000, "n_features": 20, "columns": ["mix"]},
        {"n_samples": 1000, "n_features": 500, "columns": ["mix", "binary"]},
    ]
    for estimator, params, data_spec in product(
        ["RandomForestRegressor", "ExtraTreesRegressor"],
        estimator_param_variants,
        regression_data,
    ):
        for columns in data_spec["columns"]:
            generation_kwargs = {
                key: value for key, value in data_spec.items() if key != "columns"
            }
            generation_kwargs.update(
                {
                    "columns": columns,
                    "noise": 0.1,
                    "n_informative": ceil(0.5 * data_spec["n_features"]),
                }
            )
            yield {
                "bench": bench,
                "algorithm": {
                    "estimator": estimator,
                    "estimator_params": {
                        "n_estimators": 64,
                        "max_features": 0.3,
                        "n_jobs": n_jobs,
                        **params,
                    },
                },
                "data": {
                    "source": "make_trees_regression_data",
                    "generation_kwargs": generation_kwargs,
                },
            }
