"""
Fit time vs test ROC AUC Pareto fronts of gradient boosting libraries
(scikit-learn's HistGradientBoosting, XGBoost, LightGBM, CatBoost) on real
classification datasets, following the approach of
https://github.com/scikit-learn/scikit-learn/pull/34935.

Each library fits the same ladder of hyperparameter settings, from a few
stumps to many wide trees, so that its points trace a front from cheap and
inaccurate to slow and close to the dataset's accuracy plateau. On the
smaller datasets, the widest settings overfit: their points are dominated,
which marks the end of the front.

The settings are matched across libraries (number of trees, leaves, learning
rate, histogram bins, minimum leaf size, L2 regularization of 0.1), with
lossguide/best-first growth everywhere. XGBoost has no minimum leaf size in
samples, and CatBoost keeps its default L2 regularization (its scale differs).
Early stopping is off, so the cost of a setting doesn't depend on the
library's stopping rule.

`gbdt_pareto_regression.py` runs the same settings on regression datasets,
scored by R2.

In the `gbdt` env, every library runs. In the other scikit-learn envs
(`sklearn-dev`, `sklearn-pypi`, ...), only HistGradientBoosting runs, to
place a scikit-learn branch or build on the same fronts.

Every library uses all physical cores (the default of scikit-learn and
LightGBM, set by the runner's wrappers for XGBoost and CatBoost).
"""
from _utils.implementations import implementations_for_pixi_env

BENCH = {"n_runs": 5, "py_spy_profiling": False, "time_limit": 900}
# HistGradientBoosting takes ~3 min per fit of the "longer" setting on covtype
# on a 16-core laptop.
BENCH_SLOW = {**BENCH, "n_runs": 3, "time_limit": 1800}
SLOW_DATASETS = {"covtype", "year_prediction_msd"}

# Categorical features: for HistGradientBoosting, XGBoost and LightGBM, native
# `category` columns capped at 20 categories plus a target-encoded copy of
# each ("hgb_target" preprocessing). Native splits over more categories
# overfit (e.g. on kddcup09_churn), while the target encoding keeps the
# signal of rare categories (e.g. on amazon_employee_access). CatBoost gets
# every category ("catboost" preprocessing) and encodes them itself.
PREPROCESSING_KIND = {
    "sklearn": "hgb_target",
    "xgboost": "hgb_target",
    "lightgbm": "hgb_target",
    "catboost": "catboost",
}

DATASETS = [
    # 45k x 16, low-cardinality categoricals.
    "bank_marketing",
    # 33k x 9, high-cardinality categoricals only.
    "amazon_employee_access",
    # 73k x 32, mixed numeric and categorical.
    "kick",
    # 50k x 212, many missing values, severe imbalance.
    "kddcup09_churn",
    # 581k x 12, 7 classes.
    "covtype",
]

# CatBoost's lossguide growth supports at most 64 leaves, hence the 63-leaf
# cap.
HP_SETTINGS = [
    {"name": "stumps", "n_estimators": 10, "max_leaf_nodes": 4, "learning_rate": 0.5},
    {"name": "fast", "n_estimators": 30, "max_leaf_nodes": 15, "learning_rate": 0.2},
    {"name": "medium", "n_estimators": 80, "max_leaf_nodes": 31, "learning_rate": 0.08},
    {"name": "wide", "n_estimators": 80, "max_leaf_nodes": 63, "learning_rate": 0.1},
    # Many small trees and a low learning rate: the best setting on noisy
    # datasets like kddcup09_churn, where wide trees overfit.
    {"name": "small_slow", "n_estimators": 300, "max_leaf_nodes": 7, "learning_rate": 0.05},
    {"name": "long", "n_estimators": 200, "max_leaf_nodes": 31, "learning_rate": 0.1},
    {"name": "long_wide", "n_estimators": 200, "max_leaf_nodes": 63, "learning_rate": 0.05},
    {"name": "longer", "n_estimators": 500, "max_leaf_nodes": 63, "learning_rate": 0.05},
]

MAX_BINS = 255
MIN_SAMPLES_LEAF = 20
# Without L2, HGB's leaf values can blow up when the hessians get tiny (e.g.
# rare classes on covtype), and the fit diverges on some seeds.
L2_REGULARIZATION = 0.1


ESTIMATORS = {
    "sklearn": {
        "classification": "HistGradientBoostingClassifier",
        "regression": "HistGradientBoostingRegressor",
    },
    "xgboost": {"classification": "XGBClassifier", "regression": "XGBRegressor"},
    "lightgbm": {"classification": "LGBMClassifier", "regression": "LGBMRegressor"},
    "catboost": {"classification": "CatBoostClassifier", "regression": "CatBoostRegressor"},
}


def _sklearn(hp: dict, task: str) -> dict:
    return {
        "max_iter": hp["n_estimators"],
        "max_leaf_nodes": hp["max_leaf_nodes"],
        "learning_rate": hp["learning_rate"],
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "l2_regularization": L2_REGULARIZATION,
        "max_bins": MAX_BINS,
        "early_stopping": False,
    }


def _xgboost(hp: dict, task: str) -> dict:
    return {
        "n_estimators": hp["n_estimators"],
        "max_leaves": hp["max_leaf_nodes"],
        "learning_rate": hp["learning_rate"],
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_depth": 0,
        # XGBoost has no minimum leaf size in samples, only a minimum hessian
        # sum. With the squared error, the hessian is 1 per sample, so this
        # is exact. For the log loss, use LightGBM's default instead.
        "min_child_weight": MIN_SAMPLES_LEAF if task == "regression" else 1e-3,
        "reg_lambda": L2_REGULARIZATION,
        "max_bin": MAX_BINS,
        "enable_categorical": True,
    }


def _lightgbm(hp: dict, task: str) -> dict:
    return {
        "n_estimators": hp["n_estimators"],
        "num_leaves": hp["max_leaf_nodes"],
        "learning_rate": hp["learning_rate"],
        "min_child_samples": MIN_SAMPLES_LEAF,
        "reg_lambda": L2_REGULARIZATION,
        "max_bin": MAX_BINS,
        "verbosity": -1,
    }


def _catboost(hp: dict, task: str) -> dict:
    return {
        "iterations": hp["n_estimators"],
        "max_leaves": hp["max_leaf_nodes"],
        "learning_rate": hp["learning_rate"],
        "grow_policy": "Lossguide",
        "depth": 16,
        "min_data_in_leaf": MIN_SAMPLES_LEAF,
        "max_bin": MAX_BINS,
        # No row subsampling, like the other libraries.
        "bootstrap_type": "No",
        # CatBoost's "Ordered" boosting doubles the work per tree.
        "boosting_type": "Plain",
        "verbose": False,
        "allow_writing_files": False,
    }


ESTIMATOR_PARAMS = {
    "sklearn": _sklearn,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
}


def _cases(implem: dict, datasets: list[str], task: str) -> list[dict]:
    library = implem["library"]
    cases = []
    for dataset in datasets:
        for hp in HP_SETTINGS:
            cases.append({
                "bench": BENCH_SLOW if dataset in SLOW_DATASETS else BENCH,
                "implementation": implem,
                "metadata": {"task": task, "hp_setting": hp["name"]},
                "algorithm": {
                    "estimator": ESTIMATORS[library][task],
                    "estimator_params": ESTIMATOR_PARAMS[library](hp, task),
                },
                "data": {"dataset": dataset, "preprocessing_kind": PREPROCESSING_KIND[library]},
            })
    return cases


def generate_task_cases(datasets: list[str], task: str) -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        # Skips sklearnex and the Array API implementations: they don't
        # implement HistGradientBoosting.
        if implem.get("device") is None and implem["library"] in ESTIMATOR_PARAMS:
            cases += _cases(implem, datasets, task)
    return cases


def generate_cases() -> list[dict]:
    return generate_task_cases(DATASETS, "classification")
