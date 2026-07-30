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
# For cases whose single fit takes minutes at multi-million-row scale
# (RandomForest doesn't scale as gracefully as histogram-based boosting) -
# still exercised, just not repeated 5x.
BENCH_SLOW = {"n_runs": 2}

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
# No preprocessing at all: tree models don't need scaling, and these
# datasets have no categorical columns to encode.
NO_PREPROCESSING = {}


def _linear_case(
    algorithm: dict, preprocessing: dict = LINEAR_PREPROCESSING, bench: dict = None
) -> dict:
    return {"algorithm": algorithm, "preprocessing": preprocessing, "bench": bench}


def _tree_case(
    algorithm: dict, preprocessing: dict = TREES_PREPROCESSING, bench: dict = None
) -> dict:
    return {"algorithm": algorithm, "preprocessing": preprocessing, "bench": bench}


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
    {
        # Environmental/forestry vertical. 581012 x 12: 10 continuous numeric
        # (elevation, slope, hillshade, distances, ...) plus 2 categorical
        # features (`Wilderness_Area` 4 cats, `Soil_Type` 40 cats)
        # reconstructed by `undo_one_hot` from their one-hot-encoded source
        # columns, so this exercises real categorical preprocessing rather
        # than scaling 44 already-binary columns. 7-class classification,
        # severely imbalanced (smallest class is 0.47% of rows), so
        # class_weight="balanced" is used for the linear case; RandomForest
        # handles the imbalance fine on its own.
        "dataset": "covtype",
        "task": "classification",
        "linear": [
            # LogisticRegression, linear/no-nystroem: test acc 0.628, bal.acc 0.754
            # (~70s/fit - splines over 10 numeric cols + encoding 2 categoricals)
            _linear_case(
                {
                    "estimator": "LogisticRegression",
                    "estimator_params": {
                        "C": 1.0,
                        "class_weight": "balanced",
                        "solver": "lbfgs",
                        "max_iter": 1000,
                    },
                },
                bench=BENCH_SLOW,
            ),
        ],
        "tree": [
            # RandomForestClassifier, trees/ordinal: test acc 0.963, bal.acc 0.921
            _tree_case({
                "estimator": "RandomForestClassifier",
                "estimator_params": {
                    "n_estimators": 100,
                    "random_state": 0,
                    "n_jobs": N_JOBS,
                },
            }),
        ],
    },
    {
        # This dataset's features are already standardized by the source
        # (means ~0/1, stds all in [0.2, 1.0]), so no scaling/preprocessing
        # is needed for the linear case either.
        "dataset": "susy",
        "task": "classification",
        "linear": [
            # LogisticRegression, no preprocessing: test acc 0.789, bal.acc 0.780, ROC AUC 0.859
            _linear_case(
                {
                    "estimator": "LogisticRegression",
                    "estimator_params": {
                        "max_iter": 200,
                        "solver": "newton-cholesky",
                    },
                },
                preprocessing=NO_PREPROCESSING,
            ),
        ],
        "tree": [
            # RandomForestClassifier, no preprocessing: test acc 0.798, bal.acc 0.790,
            # ROC AUC 0.869. ~250s/fit at 4.5M rows (RandomForest doesn't scale as
            # gracefully as histogram-based boosting) - kept small and run fewer times.
            _tree_case(
                {
                    "estimator": "RandomForestClassifier",
                    "estimator_params": {
                        "n_estimators": 50,
                        "max_depth": 12,
                        "random_state": 0,
                        "n_jobs": N_JOBS,
                    },
                },
                preprocessing=NO_PREPROCESSING,
                bench=BENCH_SLOW,
            ),
        ],
    },
    {
        # Music/audio vertical. 515345 x 90, all numeric, regression
        # (predict a song's release year from timbre features). Inherently
        # hard task - R2 ~0.2-0.3 matches published baselines for this
        # dataset, not an undertuned model.
        "dataset": "year_prediction_msd",
        "task": "regression",
        "linear": [
            # Ridge(alpha=1.0), linear/no-nystroem: test R2 0.266. Splines
            # place knots per-column, so they don't need prior scaling
            # despite the raw features spanning wildly different scales
            # (stds range from ~6 to ~1749 across columns).
            _linear_case({"estimator": "Ridge", "estimator_params": {"alpha": 1.0}}),
        ],
        "tree": [
            # RandomForestRegressor, no preprocessing: test R2 0.242, ~75s/fit
            # (kept small - 100 trees/depth 16 took 441s for barely better R2)
            _tree_case(
                {
                    "estimator": "RandomForestRegressor",
                    "estimator_params": {
                        "n_estimators": 30,
                        "max_depth": 10,
                        "random_state": 0,
                        "n_jobs": N_JOBS,
                    },
                },
                preprocessing=NO_PREPROCESSING,
                bench=BENCH_SLOW,
            ),
        ],
    },
    {
        # Finance/fraud vertical. 284807 x 30 (PCA-anonymized transaction
        # features), all numeric. Extreme imbalance (0.17% fraud) - much
        # more severe than kddcup09_churn's 7.3%, so class_weight="balanced"
        # matters even more here. The loader flags its V1-V28 PCA columns as
        # linear-preprocessing passthrough (already well-conditioned, not
        # spline-expanded) and spline-encodes Time_of_day/Amount with 20
        # knots instead of the default 10 - see `load_fraud`'s docstring and
        # `data_desc["preprocessing_defaults"]`.
        "dataset": "fraud",
        "task": "classification",
        "linear": [
            # LogisticRegression, linear/no-nystroem: test acc 0.946, bal.acc 0.939, ROC AUC 0.984
            _linear_case({
                "estimator": "LogisticRegression",
                "estimator_params": {"class_weight": "balanced", "max_iter": 1000},
            }),
        ],
        "tree": [
            # RandomForestClassifier, no preprocessing: test acc 1.000 (rounded),
            # bal.acc 0.860, ROC AUC 0.969
            _tree_case(
                {
                    "estimator": "RandomForestClassifier",
                    "estimator_params": {
                        "n_estimators": 200,
                        "class_weight": "balanced",
                        "random_state": 0,
                        "n_jobs": N_JOBS,
                    },
                },
                preprocessing=NO_PREPROCESSING,
            ),
        ],
    },
    {
        # Healthcare vertical. 163065 x 11, regression (predict average
        # hospital payment per diagnosis-related group/provider). Standard
        # benchmark for very-high-cardinality categorical encoding: several
        # provider-identity columns have ~2000-3300 categories each.
        "dataset": "medical_charges_nominal",
        "task": "regression",
        "linear": [
            # Ridge(alpha=1.0), linear/no-nystroem: test R2 0.982
            _linear_case({"estimator": "Ridge", "estimator_params": {"alpha": 1.0}}),
        ],
        "tree": [
            # RandomForestRegressor, trees/ordinal: test R2 0.982
            _tree_case({
                "estimator": "RandomForestRegressor",
                "estimator_params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "max_features": 0.5,
                    "random_state": 0,
                    "n_jobs": N_JOBS,
                },
            }),
        ],
    },
    {
        # Banking/finance vertical (distinct from `fraud`'s transaction-fraud
        # angle: this is telemarketing conversion). 45211 x 16, 9 low-
        # cardinality categoricals (2-12 categories) + numeric. Moderate
        # imbalance (~11.7% positive).
        "dataset": "bank_marketing",
        "task": "classification",
        "linear": [
            # LogisticRegression, linear/no-nystroem: test acc 0.838, bal.acc 0.843, ROC AUC 0.913
            _linear_case({
                "estimator": "LogisticRegression",
                "estimator_params": {
                    "C": 1.57,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                },
            }),
        ],
        "tree": [
            # RandomForestClassifier, trees/ordinal: test acc 0.900, bal.acc 0.776, ROC AUC 0.924
            _tree_case({
                "estimator": "RandomForestClassifier",
                "estimator_params": {
                    "n_estimators": 300,
                    "max_depth": 20,
                    "max_features": 0.3,
                    "min_samples_leaf": 2,
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
                        "bench": entry["bench"] or BENCH,
                        "implementation": SKLEARN_IMPLEMENTATION,
                        "metadata": {"task": spec["task"], "model_kind": kind},
                        "algorithm": entry["algorithm"],
                        "data": {"dataset": spec["dataset"], **entry["preprocessing"]},
                    }
                )
    return cases
