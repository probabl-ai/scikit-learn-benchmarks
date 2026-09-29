"""
Tree models of `sklearn.tree` and the ensembles built on them (not HGB):

- the RandomForest/ExtraTrees cases of `_synthetic_trees.py`,
- DecisionTree* and GradientBoosting* cases, with library-default
  hyperparameters, on the same synthetic data (one case per dataset),
- the RandomForest/ExtraTrees cases of `_real_datasets.py`.

ExtraTrees use random splits, which don't sort feature values: they are a
useful control when benchmarking changes to how best splits are found.
"""
from copy import deepcopy

from _utils.implementations import implementations_for_pixi_env

from _synthetic_trees import generate_cases as generate_synthetic_tree_cases
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

# Estimators benchmarked on the data of each RandomForest synthetic case.
OTHER_TREE_ESTIMATORS = {
    "RandomForestClassifier": ["DecisionTreeClassifier", "GradientBoostingClassifier"],
    "RandomForestRegressor": ["DecisionTreeRegressor", "GradientBoostingRegressor"],
}

# Parameters of the synthetic forest cases that only set the forest size
# and parallelism (see `get_estimator_params_variants`), i.e. the variant with
# default tree hyperparameters.
FOREST_ONLY_PARAMS = {"n_estimators", "max_features", "n_jobs"}


def _other_tree_cases(forest_cases: list[dict]) -> list[dict]:
    cases = []
    for case in forest_cases:
        algorithm = case["algorithm"]
        if algorithm["estimator"] not in OTHER_TREE_ESTIMATORS:
            continue
        if set(algorithm["estimator_params"]) != FOREST_ONLY_PARAMS:
            continue
        for estimator in OTHER_TREE_ESTIMATORS[algorithm["estimator"]]:
            other_case = deepcopy(case)
            other_case["algorithm"] = {"estimator": estimator, "estimator_params": {}}
            cases.append(other_case)
    return cases


def generate_cases() -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        synthetic_cases = generate_synthetic_tree_cases(implem, tier="normal")
        cases += synthetic_cases
        cases += _other_tree_cases(synthetic_cases)
        cases += [
            case
            for case in generate_real_cases(implem, max_tier="normal")
            if case["algorithm"]["estimator"] in FOREST_ESTIMATORS
        ]

    cases = list(filter_gpu_cases_if_unavailable(cases))
    cases = list(filter_unsupported_cases(cases))
    return cases
