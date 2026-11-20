"""
Tree models of `sklearn.tree` and the ensembles built on them (not HGB), sized
to run in under 1h on each runner with scikit-learn `main` (intel-gnr being
the slowest, as forests have `2 * cpu_count()` trees there).

Synthetic data (`_synthetic_trees.py`, scales 20 and 100, without the trivial
binary 1-feature data):

- RandomForest: default and `max_leaf_nodes` variants at scale 100, default
  variant at scale 20.
- ExtraTrees: default variant at scale 100.
- DecisionTree*, library defaults: on the data of every RandomForest case.
- GradientBoosting*, library defaults: at scale 20, and on the 20 features data
  at scale 100 (fitting 100 trees on the larger data takes minutes).
- ExtraTree*, library defaults: at scale 100.

Real datasets (`_real_datasets.py`):

- the RandomForest/ExtraTrees cases,
- DecisionTree*, library defaults, on all datasets but susy (4.5M samples),
- GradientBoosting*, library defaults, on the smallest datasets, with missing
  values imputed (`GradientBoosting*` don't support them on `main`),
- ExtraTree*, library defaults, on the largest datasets but susy.

ExtraTree(s) use random splits, which don't sort feature values: they are a
control group when benchmarking changes to how best splits are found.

Every repeat runs in its own subprocess (`subprocess_per_repeat`).
"""
from copy import deepcopy

from _utils.implementations import implementations_for_pixi_env

from _synthetic_trees import _synthetic_tree_cases
from _real_datasets import generate_cases as generate_real_cases

from sklbench.config.utils import (
    filter_unsupported_cases,
    filter_gpu_cases_if_unavailable,
)


FOREST_ESTIMATORS = {
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
}

# Parameters of the synthetic forest cases that only set the forest size and
# parallelism (see `get_estimator_params_variants`): the variant with default
# tree hyperparameters.
FOREST_ONLY_PARAMS = {"n_estimators", "max_features", "n_jobs"}

# (estimator prefix, scale) -> hyperparameter variants to keep, identified by
# the parameter they add on top of FOREST_ONLY_PARAMS (None for the default).
FOREST_VARIANTS = {
    ("RandomForest", 20): {None},
    ("RandomForest", 100): {None, "max_leaf_nodes"},
    ("ExtraTrees", 100): {None},
}

# Real datasets of the other tree models, by estimator prefix.
REAL_DATASETS = {
    "DecisionTree": {
        "amazon_employee_access",
        "ames_housing",
        "bank_marketing",
        "california_housing",
        "covtype",
        "fraud",
        "kddcup09_churn",
        "kick",
        "medical_charges_nominal",
        "year_prediction_msd",
    },
    "GradientBoosting": {
        "amazon_employee_access",
        "ames_housing",
        "bank_marketing",
        "california_housing",
        "kick",
        "medical_charges_nominal",
    },
    "ExtraTree": {"covtype", "fraud", "year_prediction_msd"},
}

N_RUNS = {"GradientBoosting": 3}
REAL_N_RUNS = 3


def _variant(params: dict) -> str | None:
    extra_params = set(params) - FOREST_ONLY_PARAMS
    return extra_params.pop() if extra_params else None


def _estimator_prefix(estimator: str) -> str:
    return estimator.removesuffix("Classifier").removesuffix("Regressor")


def _with_estimator(case: dict, estimator: str, n_runs: int | None) -> dict:
    """Copy of `case` with another estimator, with library defaults."""
    case = deepcopy(case)
    case["algorithm"] = {"estimator": estimator, "estimator_params": {}}
    if n_runs is not None:
        case["bench"]["n_runs"] = n_runs
    return case


def _synthetic_cases(implem: dict) -> list[dict]:
    cases = []
    for scale in (20, 100):
        for case in _synthetic_tree_cases(implem, scale=scale):
            generation_kwargs = case["data"]["generation_kwargs"]
            if (
                generation_kwargs["n_features"] == 1
                and generation_kwargs["columns"] == "binary"
            ):
                continue
            estimator = case["algorithm"]["estimator"]
            prefix = _estimator_prefix(estimator)
            variant = _variant(case["algorithm"]["estimator_params"])
            if variant in FOREST_VARIANTS.get((prefix, scale), set()):
                cases.append(case)
            if prefix != "RandomForest" or variant is not None:
                continue
            # Other tree models on the data of the default RandomForest case.
            task = estimator.removeprefix(prefix)
            other_prefixes = ["DecisionTree"]
            if scale == 20 or generation_kwargs["n_features"] == 20:
                other_prefixes.append("GradientBoosting")
            if scale == 100:
                other_prefixes.append("ExtraTree")
            for other_prefix in other_prefixes:
                cases.append(
                    _with_estimator(
                        case, other_prefix + task, N_RUNS.get(other_prefix)
                    )
                )
    return cases


def _real_cases(implem: dict) -> list[dict]:
    forest_cases = [
        case
        for case in generate_real_cases(implem, max_tier="normal")
        if case["algorithm"]["estimator"] in FOREST_ESTIMATORS
    ]
    cases = list(forest_cases)
    # One case per (dataset, task) for the other tree models, on the data of
    # the forest case.
    data_cases = {}
    for case in forest_cases:
        task = case["metadata"]["task"]
        data_cases.setdefault((case["data"]["dataset"], task), case)
    for (dataset, task), case in data_cases.items():
        suffix = "Classifier" if task == "classification" else "Regressor"
        for prefix, datasets in REAL_DATASETS.items():
            if dataset not in datasets:
                continue
            other_case = _with_estimator(case, prefix + suffix, REAL_N_RUNS)
            if prefix == "GradientBoosting":
                other_case["data"]["preprocessing_kwargs"] = {"remove_nans": True}
            cases.append(other_case)
    return cases


def generate_cases() -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        cases += _synthetic_cases(implem)
        cases += _real_cases(implem)

    for case in cases:
        case["bench"] = {**case["bench"], "subprocess_per_repeat": True}

    cases = list(filter_gpu_cases_if_unavailable(cases))
    cases = list(filter_unsupported_cases(cases))
    return cases
