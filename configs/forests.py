"""
RandomForest and ExtraTrees fitting, the multi-threaded cases of the trees
config of #126 (its single-threaded DecisionTree*, GradientBoosting* and
ExtraTree* cases are left out).

Synthetic data (`_synthetic_trees.py`, scales 20 and 100, without the trivial
binary 1-feature data):

- RandomForest: default and `max_leaf_nodes` variants at scale 100, default
  variant at scale 20.
- ExtraTrees: default variant at scale 100.

Real datasets: the RandomForest/ExtraTrees cases of `_real_datasets.py`.

Every repeat runs in its own subprocess (`subprocess_per_repeat`).
"""
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


def _variant(params: dict) -> str | None:
    extra_params = set(params) - FOREST_ONLY_PARAMS
    return extra_params.pop() if extra_params else None


def _estimator_prefix(estimator: str) -> str:
    return estimator.removesuffix("Classifier").removesuffix("Regressor")


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
            prefix = _estimator_prefix(case["algorithm"]["estimator"])
            variant = _variant(case["algorithm"]["estimator_params"])
            if variant in FOREST_VARIANTS.get((prefix, scale), set()):
                cases.append(case)
    return cases


def _real_cases(implem: dict) -> list[dict]:
    return [
        case
        for case in generate_real_cases(implem, max_tier="normal")
        if case["algorithm"]["estimator"] in FOREST_ESTIMATORS
    ]


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
