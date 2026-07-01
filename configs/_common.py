from __future__ import annotations

import os
from copy import deepcopy
from itertools import product
from math import ceil
from typing import Iterable

from joblib import cpu_count


def _merge_dicts(first: dict, second: dict) -> dict:
    result = deepcopy(first)
    for key, value in second.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _case(
    *,
    bench: dict,
    algorithm: dict,
    data: dict,
    implementation: dict | None = None,
) -> dict:
    return {
        "bench": deepcopy(bench),
        "algorithm": deepcopy(algorithm),
        "data": deepcopy(data),
        "implementation": deepcopy(implementation or {}),
    }


def with_implementations(cases: Iterable[dict], implementations: Iterable[dict]):
    return [
        _merge_dicts(case, {"implementation": implementation})
        for case, implementation in product(cases, implementations)
    ]


def exclude_estimators(cases: Iterable[dict], estimators: set[str]) -> list[dict]:
    return [
        case
        for case in cases
        if case["algorithm"].get("estimator") not in estimators
    ]


def _template_name() -> str:
    return os.environ.get("SKBENCH_MODELS_TEMPLATE", "test")


def sklearn_implementation() -> list[dict]:
    return [{"library": "sklearn"}]


def sklearnex_cpu_implementation() -> list[dict]:
    return [
        {
            "library": "sklearnex",
            "device": "cpu",
            "sklearnex_context": {
                "allow_fallback_to_host": False,
                "allow_sklearn_after_onedal": False,
            },
        }
    ]


def sklearnex_gpu_implementation() -> list[dict]:
    return [
        {
            "library": "sklearnex",
            "device": "gpu",
            "data_library": "dpnp",
            "sklearnex_context": {
                "array_api_dispatch": True,
                "allow_fallback_to_host": False,
                "allow_sklearn_after_onedal": False,
            },
        }
    ]


def array_api_cpu_implementations() -> list[dict]:
    return [
        {
            "library": "sklearn",
            "device": "cpu",
            "data_library": "torch",
            "sklearn_context": {"array_api_dispatch": True},
        }
    ]


def array_api_intel_implementations() -> list[dict]:
    return [
        {
            "library": "sklearn",
            "device": "xpu",
            "data_library": "torch",
            "sklearn_context": {"array_api_dispatch": True},
        },
        {
            "library": "sklearn",
            "device": "gpu",
            "data_library": "dpnp",
            "sklearn_context": {"array_api_dispatch": True},
        },
    ]


def array_api_nvidia_implementations() -> list[dict]:
    return [
        {
            "library": "sklearn",
            "device": "cuda",
            "data_library": "torch",
            "sklearn_context": {"array_api_dispatch": True},
        },
        {
            "library": "sklearn",
            "device": "cuda",
            "data_library": "cupy",
            "sklearn_context": {"array_api_dispatch": True},
        },
    ]


def tree_cases(template: str | None = None) -> list[dict]:
    template = template or _template_name()
    return _fast_tree_cases() if template == "fast" else _test_tree_cases()


def linear_cases(template: str | None = None) -> list[dict]:
    template = template or _template_name()
    return _linear_cases(template, array_api=False)


def linear_array_api_cases(template: str | None = None) -> list[dict]:
    template = template or _template_name()
    return _linear_cases(template, array_api=True)


def clustering_cases(template: str | None = None) -> list[dict]:
    template = template or _template_name()
    return _fast_clustering_cases() if template == "fast" else _test_clustering_cases()


def apply_sklearnex_cpu_tree_variants(cases: Iterable[dict]) -> list[dict]:
    result = []
    for case in cases:
        result.append(deepcopy(case))
        n_samples = case["data"].get("generation_kwargs", {}).get("n_samples")
        if n_samples is None:
            continue
        result.append(
            _merge_dicts(
                case,
                {"algorithm": {"estimator_params": {"max_bins": n_samples}}},
            )
        )
    return result


def _test_tree_cases() -> list[dict]:
    bench = {"n_runs": 3}
    base_algorithm = {
        "estimator_params": {
            "n_estimators": 16,
            "max_features": 0.5,
            "max_depth": 4,
            "n_jobs": 1,
        },
    }
    base_generation = {
        "n_samples": 1000,
        "n_features": 10,
        "n_informative": 5,
    }
    split_kwargs = {"test_size": 0.2}
    cases = []
    for estimator in ["RandomForestClassifier", "ExtraTreesClassifier"]:
        cases.append(
            _case(
                bench=bench,
                algorithm=_merge_dicts(base_algorithm, {"estimator": estimator}),
                data={
                    "source": "make_trees_classification_data",
                    "generation_kwargs": _merge_dicts(
                        base_generation,
                        {"columns": "mix", "n_classes": 2},
                    ),
                    "split_kwargs": split_kwargs,
                },
            )
        )
    for estimator in ["RandomForestRegressor", "ExtraTreesRegressor"]:
        cases.append(
            _case(
                bench=bench,
                algorithm=_merge_dicts(base_algorithm, {"estimator": estimator}),
                data={
                    "source": "make_trees_regression_data",
                    "generation_kwargs": _merge_dicts(
                        base_generation,
                        {"columns": "mix", "noise": 0.1},
                    ),
                    "split_kwargs": split_kwargs,
                },
            )
        )
    return cases


def _fast_tree_cases() -> list[dict]:
    bench = {"n_runs": 5, "time_limit": 10}
    base_algorithm = {
        "estimator_params": {
            "n_estimators": 64,
            "max_features": 0.3,
            "n_jobs": cpu_count(only_physical_cores=True),
        },
    }
    estimator_param_variants = [{}, {"max_depth": 4}, {"max_leaf_nodes": 1000}]
    split_kwargs = {"test_size": 0.2}
    cases = []

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
            cases.append(
                _case(
                    bench=bench,
                    algorithm=_merge_dicts(
                        base_algorithm,
                        {
                            "estimator": estimator,
                            "estimator_params": params,
                        },
                    ),
                    data={
                        "source": "make_trees_classification_data",
                        "generation_kwargs": generation_kwargs,
                        "split_kwargs": split_kwargs,
                    },
                )
            )

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
            cases.append(
                _case(
                    bench=bench,
                    algorithm=_merge_dicts(
                        base_algorithm,
                        {
                            "estimator": estimator,
                            "estimator_params": params,
                        },
                    ),
                    data={
                        "source": "make_trees_regression_data",
                        "generation_kwargs": generation_kwargs,
                        "split_kwargs": split_kwargs,
                    },
                )
            )
    return cases


def _linear_cases(template: str, *, array_api: bool) -> list[dict]:
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

    base_algorithm = {}
    split_kwargs = {"test_size": 0.2}
    cases = []
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
            "generation_kwargs": _merge_dicts(
                generation_kwargs, extra_generation_kwargs
            ),
            "split_kwargs": split_kwargs,
        }
        if not array_api or algorithm["estimator"] == "Ridge":
            data["order"] = "C"
        cases.append(
            _case(
                bench=bench,
                algorithm=_merge_dicts(base_algorithm, algorithm),
                data=data,
            )
        )
    return cases


def _test_clustering_cases() -> list[dict]:
    generation_kwargs = {
        "centers": 5,
        "cluster_std": 1.0,
        "n_samples": 2000,
        "n_features": 20,
    }
    return [
        _case(
            bench={"n_runs": 3},
            algorithm={
                "estimator": "KMeans",
                "estimator_params": {
                    "n_clusters": generation_kwargs["centers"],
                    "n_init": 1,
                    "max_iter": 30,
                    "tol": 0.001,
                },
            },
            data={
                "source": "make_blobs",
                "generation_kwargs": generation_kwargs,
                "split_kwargs": {"ignore": True},
            },
        )
    ]


def _fast_clustering_cases() -> list[dict]:
    bench = {"n_runs": 5, "time_limit": 10}
    base_algorithm = {"estimator": "KMeans"}
    params = {"n_init": 1, "max_iter": 30, "tol": 0.001}
    return [
        _case(
            bench=bench,
            algorithm=_merge_dicts(
                base_algorithm,
                {
                    "estimator_params": _merge_dicts(
                        params,
                        {"n_clusters": 20},
                    )
                },
            ),
            data={
                "source": "make_blobs",
                "generation_kwargs": {
                    "centers": 20,
                    "cluster_std": 0.5,
                    "n_samples": 500000,
                    "n_features": 3,
                },
                "split_kwargs": {"ignore": True},
            },
        ),
        _case(
            bench=bench,
            algorithm=_merge_dicts(
                base_algorithm,
                {"estimator_params": _merge_dicts(params, {"n_clusters": 10})},
            ),
            data={
                "dataset": "mnist",
                "split_kwargs": {"train_size": 50000, "test_size": None},
                "preprocessing_kwargs": {"normalize": "minmax"},
            },
        ),
        _case(
            bench=bench,
            algorithm=_merge_dicts(
                base_algorithm,
                {"estimator_params": _merge_dicts(params, {"n_clusters": 100})},
            ),
            data={
                "dataset": "mnist",
                "split_kwargs": {"train_size": 10000, "test_size": 10000},
                "preprocessing_kwargs": {"normalize": "minmax"},
            },
        ),
    ]
