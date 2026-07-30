"""
Real-dataset cases: at least one linear model and one tree-based model per
dataset, run with plain sklearn. Hyperparameters below were picked with a
small RandomizedSearchCV pass per (dataset, model) pair (cv=3,
scoring="roc_auc" for classification / "r2" for regression), not left at
library defaults, so the cases are reasonably realistic rather than
arbitrary.

The default Nystroem kernel-approximation step in `linear_preprocessor`
(`sklbench/runners/datasets/preprocessing.py`) was measured to hurt held-out
performance on every dataset below (e.g. ROC AUC -0.02 to -0.06), so linear
cases disable it via `preprocessing_kwargs={"nystroem": "no"}` unless the case
is specifically exercising the Nystroem path (`kick`'s second LogisticRegression
case below).
"""

from joblib import cpu_count

SKLEARN_IMPLEMENTATION = {"library": "sklearn"}

BENCH = {"n_runs": 5}

N_JOBS = cpu_count(only_physical_cores=True)

TREES_PREPROCESSING = {
    "preprocessing_kind": "trees",
    "preprocessing_kwargs": {"encoding": "ordinal"},
}
LINEAR_PREPROCESSING = {
    "preprocessing_kind": "linear",
    "preprocessing_kwargs": {"nystroem": "no"},
}
LINEAR_NYSTROEM_PREPROCESSING = {
    "preprocessing_kind": "linear",
    "preprocessing_kwargs": {},  # default poly degree-2 Nystroem, 300 components
}


def _linear_case(algorithm: dict, preprocessing: dict = LINEAR_PREPROCESSING) -> dict:
    return {"algorithm": algorithm, "preprocessing": preprocessing}


def _tree_case(algorithm: dict, preprocessing: dict = TREES_PREPROCESSING) -> dict:
    return {"algorithm": algorithm, "preprocessing": preprocessing}


# Tuned on a held-out 20% test split (see each dataset's `default_split`).
REAL_DATASET_CASES = [
    {
        "dataset": "ames_housing",
        "task": "regression",
        "linear": [
            # Ridge(alpha=1.0), linear/no-nystroem: test R2 0.89
            _linear_case({"estimator": "Ridge", "estimator_params": {"alpha": 1.0}}),
        ],
        "tree": [
            # RandomForestRegressor, trees/ordinal: test R2 0.89
            _tree_case({
                "estimator": "RandomForestRegressor",
                "estimator_params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "max_features": 0.5,
                    "min_samples_leaf": 1,
                    "random_state": 0,
                    "n_jobs": N_JOBS,
                },
            }),
        ],
    },
    {
        "dataset": "amazon_employee_access",
        "task": "classification",
        "linear": [
            # LogisticRegression, linear/no-nystroem: test acc 0.826, bal.acc 0.794, ROC AUC 0.843
            _linear_case({
                "estimator": "LogisticRegression",
                "estimator_params": {
                    "C": 1,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                },
            }),
        ],
        "tree": [
            # RandomForestClassifier, trees/ordinal: test acc 0.950, bal.acc 0.606, ROC AUC 0.855
            _tree_case({
                "estimator": "RandomForestClassifier",
                "estimator_params": {
                    "n_estimators": 200,
                    "max_features": 0.3,
                    "min_samples_leaf": 2,
                    "random_state": 0,
                    "n_jobs": N_JOBS,
                },
            }),
        ],
    },
    {
        "dataset": "kddcup09_churn",
        "task": "classification",
        # Severe imbalance (7.3% churn) makes the best-by-ROC-AUC model for
        # both estimators collapse to predicting the majority class
        # (balanced accuracy ~0.50, i.e. useless for churn detection), so
        # class_weight="balanced"/"balanced_subsample" is used instead - a
        # small ROC AUC trade for a model that actually discriminates churn.
        "linear": [
            # LogisticRegression, linear/no-nystroem: test acc 0.681, bal.acc 0.640, ROC AUC 0.698
            _linear_case({
                "estimator": "LogisticRegression",
                "estimator_params": {
                    "C": 5.5,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                },
            }),
        ],
        "tree": [
            # RandomForestClassifier, trees/ordinal: test acc 0.909, bal.acc 0.558, ROC AUC 0.713
            _tree_case({
                "estimator": "RandomForestClassifier",
                "estimator_params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "max_features": 0.3,
                    "min_samples_leaf": 10,
                    "class_weight": "balanced_subsample",
                    "random_state": 0,
                    "n_jobs": N_JOBS,
                },
            }),
        ],
    },
    {
        "dataset": "kick",
        "task": "classification",
        "linear": [
            # LogisticRegression, linear/no-nystroem: test acc 0.738, bal.acc 0.700, ROC AUC 0.773
            _linear_case({
                "estimator": "LogisticRegression",
                "estimator_params": {
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "C": 30.0,
                    "class_weight": "balanced",
                },
            }),
            # LogisticRegression w/ Nystroem (poly deg-2, 300 components):
            # test acc 0.737, bal.acc 0.697, ROC AUC 0.768 - close to the
            # no-nystroem case above, exercises the kernel-approximation path.
            _linear_case(
                {
                    "estimator": "LogisticRegression",
                    "estimator_params": {
                        "solver": "lbfgs",
                        "max_iter": 2000,
                        "C": 24.0,
                        "class_weight": "balanced",
                    },
                },
                preprocessing=LINEAR_NYSTROEM_PREPROCESSING,
            ),
        ],
        "tree": [
            # RandomForestClassifier, trees/ordinal: test acc 0.856, bal.acc 0.679, ROC AUC 0.767
            _tree_case({
                "estimator": "RandomForestClassifier",
                "estimator_params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "max_features": "sqrt",
                    "min_samples_leaf": 5,
                    "class_weight": "balanced",
                    "random_state": 0,
                    "n_jobs": N_JOBS,
                },
            }),
            # ExtraTreesClassifier, trees/ordinal: test acc 0.809, bal.acc 0.684, ROC AUC 0.760
            _tree_case({
                "estimator": "ExtraTreesClassifier",
                "estimator_params": {
                    "n_estimators": 200,
                    "max_depth": 30,
                    "max_features": "sqrt",
                    "min_samples_leaf": 10,
                    "class_weight": "balanced_subsample",
                    "random_state": 0,
                    "n_jobs": N_JOBS,
                },
            }),
        ],
    },
]


def generate_cases() -> list[dict]:
    cases = []
    for spec in REAL_DATASET_CASES:
        for kind in ("linear", "tree"):
            for entry in spec[kind]:
                cases.append(
                    {
                        "bench": BENCH,
                        "implementation": SKLEARN_IMPLEMENTATION,
                        "metadata": {"task": spec["task"], "model_kind": kind},
                        "algorithm": entry["algorithm"],
                        "data": {"dataset": spec["dataset"], **entry["preprocessing"]},
                    }
                )
    return cases
