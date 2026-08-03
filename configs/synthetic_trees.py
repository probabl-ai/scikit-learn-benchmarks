from copy import deepcopy
from itertools import product, chain
from typing import Iterable

from joblib import cpu_count
from math import floor

N_JOBS = floor(0.9 * cpu_count(only_physical_cores=True))

from _common import deterministic_random_choice


def _stable_seed(case: dict) -> dict:
    """Strip fields that vary across otherwise-identical cases so
    `deterministic_random_choice` picks the same cases regardless of which
    machine or implementation generated them:

    - `case["algorithm"]["estimator_params"]` embeds `n_jobs`/`n_estimators`,
      both scaled from `N_JOBS = joblib.cpu_count(...)`, i.e. machine-dependent,
      plus `max_bins`, which is only ever added for sklearnex cases.
    - `case["implementation"]` differs between sklearn/sklearnex/array-API for
      what is meant to be the same logical case. Hashing it in means each
      implementation independently samples a different one-third subset of
      the matrix, so a case kept for sklearn can be dropped for sklearnex (and
      vice versa) - breaking the sklearn-vs-sklearnex/array-API comparison for
      most of the matrix.
    """
    seed = deepcopy(case)
    seed["algorithm"]["estimator_params"].pop("n_jobs", None)
    seed["algorithm"]["estimator_params"].pop("n_estimators", None)
    seed["algorithm"]["estimator_params"].pop("max_bins", None)
    seed.pop("implementation", None)
    return seed


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
    bench = {"n_runs": 5, "time_limit": 2 + scale * 2}

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
                and deterministic_random_choice(_stable_seed(case), [True, False])
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
        if deterministic_random_choice(_stable_seed(case), [0, 0, 1])
    ]

    return cases
