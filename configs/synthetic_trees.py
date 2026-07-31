from copy import deepcopy
from itertools import product, chain
from typing import Iterable

from joblib import cpu_count

N_JOBS = cpu_count(only_physical_cores=True)

from _common import deterministic_random_choice


def get_estimator_params_variants(n_samples: int, n_estimators: int | None = None, broad=False):
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
        "n_estimators": n_estimators or N_JOBS * 5,
        "max_features": 0.3,
        "n_jobs": N_JOBS
    }

    for params in params_list:
        yield {**params, **params_base}


def _synthetic_tree_cases(implem: dict, scale: int = 10) -> Iterable[dict]:
    bench = {"n_runs": 5, "time_limit": scale}

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
            if (
                implem["library"] == "sklearnex"
                and deterministic_random_choice(case, [True, False])
            ):
                case["algorithm"]["estimator_params"]["max_bins"] = n_samples
            yield case


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

    if tier == "fast":
        return list(_synthetic_tree_cases(implem, scale=10))

    cases = list(chain(
        _synthetic_tree_cases(implem, scale=10),
        _synthetic_tree_cases(implem, scale=20),
        _synthetic_tree_cases(implem, scale=50),
        _synthetic_tree_cases(implem, scale=100),
    ))

    if tier == "slow":
        cases += list(chain(
            _synthetic_tree_cases(implem, scale=200),
            _synthetic_tree_cases(implem, scale=500),
            _synthetic_tree_cases(implem, scale=1000),
        ))

    # sub-sample one third of the matrix
    cases = [
        case for case in cases
        if deterministic_random_choice(case, [0, 0, 1])
    ]

    return cases
