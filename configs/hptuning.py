"""
Joblib-level-parallelism study: `RandomizedSearchCV` over a representative
slice of `real_datasets.py`'s datasets, each paired with the same linear +
tree-ensemble estimator `real_datasets.py` picked for it, plus its
HistGradientBoosting counterpart (sklearn only - sklearnex has no
accelerated HGB). Every case leaves `hptuning.outer_n_jobs`/`inner_n_jobs`
unset, so the point isn't a hand-tuned parallelism split but the *default*
one: outer defaults to `round(sqrt(physical_cores))` concurrent
RandomizedSearchCV candidates (see
`sklbench.config.models.hptuning.resolve_outer_n_jobs`), and each candidate
is left to whatever its library defaults to on its own - RF/ET's own
`n_jobs=1`, HGB's OpenMP thread pool, BLAS threads under
Ridge/LogisticRegression - rather than something explicitly balanced
against the outer level. That's the behavior worth measuring on a big
many-core box (e.g. a GNR): whether those defaults compose into reasonable
core usage or start oversubscribing.

Tree-ensemble `n_estimators` is fixed (not scaled by core count like
`real_datasets.py`'s `N_JOBS * k`) precisely because `n_jobs` is left
unset here - at 1 thread per candidate, a core-count-scaled forest would
make a single candidate fit take as long as the whole `all_models.py` case
it's drawn from. `max_samples` similarly bounds the larger datasets so a
10-candidate x 3-fold search stays cheap even when only one candidate runs
at a time.

Real datasets are the 6 `hgb_scalability.py`'s `REAL_SCALING_DATASETS`
already picked to span `real_datasets.py`'s size/categorical-content range,
so this matrix stays a subsample of `all_models.py`'s rather than repeating
its full dataset list. A handful of `make_classification`/`make_regression`
synthetic shapes (narrow/wide, one pair per task) are mixed in alongside
them, sized so a single search run lands in the same ~10s ballpark as the
real-dataset cases above - see `SYNTHETIC_SPECS` for why that needs two
differently-shaped datasets per spec rather than one shared shape.
"""

import numpy as np
import math

from joblib import cpu_count
from _implementations import implementations_for_pixi_env

from sklbench.config import Algorithm, Data, HPTuning, HPTuningCase


def sqrt_physical_cores(hptuning: HPTuning) -> int:
    if hptuning.n_jobs is not None:
        return hptuning.n_jobs
    physical_cores = round(cpu_count(only_physical_cores=True))
    return max(1, round(math.sqrt(physical_cores)))


BENCH = {"n_runs": 3}

# Keys are full pipeline param paths - "estimator__..." here since these
# tune the model itself, not preprocessing (see the runner's
# `_build_pipeline` for the "preprocessor"/"estimator" step names).
_HGB_PARAM_DISTRIBUTIONS = {
    "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    "estimator__max_leaf_nodes": [7, 15, 31, 63],
    "estimator__min_samples_leaf": [5, 10, 20, 50],
    "estimator__l2_regularization": [0.0, 0.1, 0.3, 1.0],
    "estimator__max_iter": [50, 100, 150],
}
_HGB_PARAMS = {"early_stopping": False}

_RIDGE_PARAM_DISTRIBUTIONS = {"estimator__alpha": list(np.logspace(-3, 3, 13))}
_RIDGE_PARAMS = {}

# Not "balanced": see `sklbench.config.utils` callers / repo notes - it
# distorts calibration without moving ROC AUC.
_LOGISTIC_REGRESSION_PARAMS = {"solver": "lbfgs", "max_iter": 1000}
_LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS = {
    "estimator__C": list(np.logspace(-3, 3, 13)),
    "estimator__fit_intercept": [True, False],
}

# Shared by RandomForest{Classifier,Regressor}/ExtraTrees{Classifier,Regressor}:
# same param names across all four.
_TREE_PARAMS = {"n_estimators": 100}
_TREE_PARAM_DISTRIBUTIONS = {
    # Capped well below "unbounded": on the wider real datasets below
    # (year_prediction_msd, 90 features), even max_depth=30 left a single RF
    # candidate running past 240s at min_samples_leaf=1 - see the module
    # docstring on cost control.
    "estimator__max_depth": [3, 6, 10, 15],
    "estimator__max_features": [0.3, 0.5, "sqrt"],
    "estimator__min_samples_leaf": [1, 5, 10, 20],
}

_ESTIMATOR_FAMILIES = {
    "Ridge": (_RIDGE_PARAMS, _RIDGE_PARAM_DISTRIBUTIONS),
    "LogisticRegression": (
        _LOGISTIC_REGRESSION_PARAMS,
        _LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
    ),
    "RandomForestClassifier": (_TREE_PARAMS, _TREE_PARAM_DISTRIBUTIONS),
    "RandomForestRegressor": (_TREE_PARAMS, _TREE_PARAM_DISTRIBUTIONS),
    "ExtraTreesClassifier": (_TREE_PARAMS, _TREE_PARAM_DISTRIBUTIONS),
    "ExtraTreesRegressor": (_TREE_PARAMS, _TREE_PARAM_DISTRIBUTIONS),
    "HistGradientBoostingClassifier": (_HGB_PARAMS, _HGB_PARAM_DISTRIBUTIONS),
    "HistGradientBoostingRegressor": (_HGB_PARAMS, _HGB_PARAM_DISTRIBUTIONS),
}

_HGB_ESTIMATORS = {
    "regression": "HistGradientBoostingRegressor",
    "classification": "HistGradientBoostingClassifier",
}

# A spec is (linear_data, tree_data, linear_estimator, tree_estimator, task,
# linear_max_samples, tree_max_samples). `tree_data`/`tree_max_samples` also
# apply to the HGB case. Real-dataset specs below share one `data`/
# `max_samples` across all three estimators (`_real_spec` duplicates it into
# both slots); the synthetic ones don't, per-family costs diverge too much
# on one shared dataset to land every estimator near the same wall time -
# see `SYNTHETIC_SPECS`.


def _real_spec(dataset: str, linear: str, tree: str, task: str, max_samples: int | None):
    data = Data(dataset=dataset)
    return (data, data, linear, tree, task, max_samples, max_samples)


# Same 6 datasets as `hgb_scalability.py`'s `REAL_SCALING_DATASETS`, each
# paired with the linear/tree estimator `real_datasets.py` uses for it there.
DATASET_SPECS = [
    _real_spec("ames_housing", "Ridge", "RandomForestRegressor", "regression", None),
    _real_spec(
        "amazon_employee_access",
        "LogisticRegression", "RandomForestClassifier", "classification", 8000,
    ),
    _real_spec(
        "kddcup09_churn",
        "LogisticRegression", "RandomForestClassifier", "classification", 6000,
    ),
    _real_spec(
        "year_prediction_msd", "Ridge", "RandomForestRegressor", "regression", 5000,
    ),
    _real_spec(
        "covtype",
        "LogisticRegression", "RandomForestClassifier", "classification", 15000,
    ),
    _real_spec(
        "susy",
        "LogisticRegression", "ExtraTreesClassifier", "classification", 15000,
    ),
]

# Narrow/wide pair per task, plain numeric (no missing values, no
# categoricals). Ridge/LogisticRegression need far more features than
# RF/ET/HGB to reach a comparable wall time (measured: ~10-12s at
# n_samples=30000/n_features=300 vs. RF at the same shape still running
# past 240s at n_samples as low as 2000 - RF/ET's per-split cost scales
# with feature count directly, unlike a single BLAS-backed lbfgs/Cholesky
# solve) - hence the separate, much narrower `tree_data` per spec below
# rather than one shared dataset.
SYNTHETIC_SPECS = [
    (
        Data(
            source="make_classification",
            generation_kwargs={
                "n_samples": 30000, "n_features": 300, "n_informative": 100, "n_classes": 2,
            },
        ),
        Data(
            source="make_classification",
            generation_kwargs={
                "n_samples": 5000, "n_features": 20, "n_informative": 10, "n_classes": 2,
            },
        ),
        "LogisticRegression", "RandomForestClassifier", "classification", None, None,
    ),
    (
        Data(
            source="make_classification",
            generation_kwargs={
                "n_samples": 20000, "n_features": 400, "n_informative": 120, "n_classes": 2,
            },
        ),
        Data(
            source="make_classification",
            generation_kwargs={
                "n_samples": 4000, "n_features": 100, "n_informative": 20, "n_classes": 2,
            },
        ),
        "LogisticRegression", "ExtraTreesClassifier", "classification", None, None,
    ),
    (
        Data(
            source="make_regression",
            generation_kwargs={
                "n_samples": 30000, "n_features": 300, "n_informative": 100, "noise": 0.1,
            },
        ),
        Data(
            source="make_regression",
            generation_kwargs={
                "n_samples": 5000, "n_features": 20, "n_informative": 10, "noise": 0.1,
            },
        ),
        "Ridge", "RandomForestRegressor", "regression", None, None,
    ),
    (
        Data(
            source="make_regression",
            generation_kwargs={
                "n_samples": 20000, "n_features": 400, "n_informative": 120, "noise": 0.1,
            },
        ),
        Data(
            source="make_regression",
            generation_kwargs={
                "n_samples": 4000, "n_features": 100, "n_informative": 30, "noise": 0.1,
            },
        ),
        "Ridge", "ExtraTreesRegressor", "regression", None, None,
    ),
]


def _case(data: Data, estimator: str, implem: dict, max_samples: int | None) -> list[HPTuningCase]:
    cases = []
    estimator_params, param_distributions = _ESTIMATOR_FAMILIES[estimator]
    n_cores = cpu_count(only_physical_cores=True)
    for n_jobs in [1, round(math.sqrt(n_cores)), n_cores // 2]:
        # at least 10 iterations, and a multiple of n_jobs:
        n_iter = min(n_jobs * k for k in range(3, 11) if n_jobs * k >= 10)
        cases.append(HPTuningCase(
            bench=BENCH,
            algorithm=Algorithm(estimator=estimator, estimator_params=estimator_params),
            data=data,
            implementation=implem,
            hptuning=HPTuning(
                param_distributions=param_distributions,
                n_iter=n_iter,
                max_samples=max_samples,
                n_jobs=n_jobs
            ),
        ))
    return cases


def generate_cases() -> list[HPTuningCase]:
    # CPU-only, non-array-API implementations: the runner's preprocessing
    # (plain numpy/pandas ColumnTransformer) isn't device- or
    # array-API-aware, and thread-count defaults aren't a meaningful axis
    # for GPU-offloaded work anyway.
    implementations = [
        implem
        for implem in implementations_for_pixi_env()
        if implem.get("data_library") is None and implem.get("device") in (None, "cpu")
    ]

    cases = []
    for (
        linear_data, tree_data, linear_estimator, tree_estimator, task,
        linear_max_samples, tree_max_samples,
    ) in DATASET_SPECS + SYNTHETIC_SPECS:
        for implem in implementations:
            cases.extend(_case(linear_data, linear_estimator, implem, linear_max_samples))
            cases.extend(_case(tree_data, tree_estimator, implem, tree_max_samples))
            if implem["library"] == "sklearn":
                cases.extend(
                    _case(tree_data, _HGB_ESTIMATORS[task], implem, tree_max_samples)
                )

    return cases
