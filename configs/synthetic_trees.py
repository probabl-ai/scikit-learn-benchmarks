from copy import deepcopy
from itertools import product, chain
from typing import Iterable
from math import sqrt

from joblib import cpu_count

from _common import deterministic_random_choice


def get_estimator_params_variants(n_samples: int, broad=False):
    params_list = [
        {},
        {"min_samples_split": round(n_samples ** 0.22)},
        {"max_leaf_nodes": round(n_samples ** 0.7 / 3)}
    ]
    if broad:
        # also include cases that are interesting to profile but
        # not representative of RF/ET usage
        params_list += [
            {"max_depth": 4},
            {"max_leaf_nodes": 32},
        ]

    params_base = {
        "n_estimators": cpu_count() * 3,
        "max_features": 0.3,
        "n_jobs": -1
    }

    for params in params_list:
        yield {**params, **params_base}


def tree_data_shapes(scale: int) -> list[dict]:
    """Base (n_samples, n_features, n_informative) shapes at a given scale.

    Shared with `all_models_scaling.py`'s thread-count scaling study so both
    configs stay in sync on what "scale N" means.
    """
    return [
        {"n_samples": 10000 * scale, "n_features": 1, "n_informative": 1},
        {"n_samples": 1000 * scale, "n_features": 20, "n_informative": 10},
        {"n_samples": 100 * scale, "n_features": 500, "n_informative": 50},
    ]


def _synthetic_tree_cases(implem: dict, scale: int = 10) -> Iterable[dict]:
    time_limit = round(
        2 + scale * (2 + sqrt(cpu_count() / 16))
    )
    bench = {
        "n_runs": 5,
        "time_limit": time_limit,
        "py_spy_profiling": False,
    }

    base_data = tree_data_shapes(scale)
    data = []
    for data_spec_mix in base_data[:]:
        data_spec_mix["columns"] = "mix"
        data_spec_other = deepcopy(data_spec_mix)
        data_spec_other["columns"] = deterministic_random_choice(
            seed=data_spec_mix,
            choices=["continuous", "binary", "long-tail"]
        )
        data.append((data_spec_mix, data_spec_other))

    classification_data = []
    regression_data = []
    for (data_spec_clf, data_spec_reg) in data:
        seed = (data_spec_clf, data_spec_reg)
        if deterministic_random_choice(seed, [0, 1]):
            data_spec_clf, data_spec_reg = data_spec_reg, data_spec_clf

        data_spec_clf.update({
            "n_classes": 2,
            "n_redundant": 0,
        })
        if data_spec_clf["n_features"] == 1:
            data_spec_clf["n_clusters_per_class"] = 1
        classification_data.append(data_spec_clf)

        data_spec_reg.update({
            "noise": 0.1,
        })
        regression_data.append(data_spec_reg)

    entries = chain(
        product(
            ["RandomForestClassifier", "ExtraTreesClassifier"],
            classification_data,
        ),
        product(
            ["RandomForestRegressor", "ExtraTreesRegressor"],
            regression_data,
        ),
    )

    cases = []
    for estimator, generation_kwargs in entries:
        n_samples = generation_kwargs["n_samples"]
        source = (
            "make_trees_classification_data"
            if estimator.endswith("Classifier")
            else "make_trees_regression_data"
        )
        for params in get_estimator_params_variants(n_samples):
            case = {
                "bench": bench,
                "algorithm": {
                    "estimator": estimator,
                    "estimator_params": params,
                },
                "data": {
                    "source": source,
                    "generation_kwargs": generation_kwargs,
                },
                "implementation": implem,
            }
            cases.append(case)
            if implem["library"] == "sklearnex":
                case = deepcopy(case)
                case["algorithm"]["estimator_params"]["max_bins"] = n_samples
                cases.append(case)        

    return cases


def _test_tree_cases(implem: dict) -> Iterable[dict]:
    cases_per_model = {}
    for case in _synthetic_tree_cases(implem, scale=2):
        cases_per_model.setdefault(case["algorithm"]["estimator"], []).append(case)

    # select one per algorithm
    for estimator, cases in cases_per_model.items():
        case = deterministic_random_choice(seed=estimator, choices=cases)
        case = deepcopy(case)
        case["bench"]["n_runs"] = 2
        yield case


def generate_cases(implem: dict | None = None, tier: str = "normal") -> list[dict]:
    if implem is None:
        implem = {"library": "sklearn"}

    if tier == "test":
        return list(_test_tree_cases(implem))

    scales = {
        "fast": [10],
        "normal": [10, 100],
        "slow": [10, 100, 1000],
    }

    cases = list(chain(*[
        _synthetic_tree_cases(implem, scale=scale)
        for scale in scales[tier]
    ]))

    return cases
