"""
Tree models of `sklearn.tree` and the ensembles built on them (not HGB), sized
to run in well under 1h on each runner (about 35 min of fit/predict with
scikit-learn `main` on intel-gnr, where forests have `2 * cpu_count()` trees):

- RandomForest cases of `_synthetic_trees.py`: all data shapes at scales 20
  and 100, with the default and `max_leaf_nodes` variants at scale 100 and
  only the default variant at scale 20.
- ExtraTrees cases of `_synthetic_trees.py`, default variant at scale 100.
  ExtraTrees use random splits, which don't sort feature values: they are a
  control when benchmarking changes to how best splits are found.
- DecisionTree* cases, with library-default hyperparameters, on the data of
  every synthetic RandomForest case above.
- GradientBoosting* cases, with library-default hyperparameters, on the same
  data at scale 20 and on the 20 features data at scale 100 (fitting 100
  trees on the 500 features data at scale 100 takes minutes).
- The RandomForest/ExtraTrees cases of `_real_datasets.py`.
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

GB_N_RUNS = 3


def _variant(params: dict) -> str | None:
    extra_params = set(params) - FOREST_ONLY_PARAMS
    return extra_params.pop() if extra_params else None


def _synthetic_cases(implem: dict) -> list[dict]:
    cases = []
    for scale in (20, 100):
        for case in _synthetic_tree_cases(implem, scale=scale):
            algorithm = case["algorithm"]
            prefix = algorithm["estimator"].removesuffix("Classifier").removesuffix(
                "Regressor"
            )
            variant = _variant(algorithm["estimator_params"])
            if variant in FOREST_VARIANTS.get((prefix, scale), set()):
                cases.append(case)
            if prefix != "RandomForest" or variant is not None:
                continue
            # Other tree models on the data of the default RandomForest case.
            task = algorithm["estimator"].removeprefix("RandomForest")
            n_features = case["data"]["generation_kwargs"]["n_features"]
            other_estimators = [f"DecisionTree{task}"]
            if scale == 20 or n_features == 20:
                other_estimators.append(f"GradientBoosting{task}")
            for estimator in other_estimators:
                other_case = deepcopy(case)
                other_case["algorithm"] = {"estimator": estimator, "estimator_params": {}}
                if estimator.startswith("GradientBoosting"):
                    other_case["bench"]["n_runs"] = GB_N_RUNS
                cases.append(other_case)
    return cases


def generate_cases() -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        cases += _synthetic_cases(implem)
        cases += [
            case
            for case in generate_real_cases(implem, max_tier="normal")
            if case["algorithm"]["estimator"] in FOREST_ESTIMATORS
        ]

    cases = list(filter_gpu_cases_if_unavailable(cases))
    cases = list(filter_unsupported_cases(cases))
    return cases
