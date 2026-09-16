"""
RandomizedSearchCV cases over a shorter, hand-picked list of real datasets
than `hptuning.py`'s DATASET_SPECS/SYNTHETIC_SPECS matrix, with an actual
search space per (dataset, estimator) pair instead of one shared search
space per estimator family - meant to produce more realistic, more varied
hptuning data points than `hptuning.py`.

Each `REAL_DATASET_CASES` entry is:

    (
        (dataset, max_samples),          # or (dataset, max_samples, preprocessing_kind)
        estimator,                       # str, or list[str] to reuse one search space
        search_space,                    # {"estimator": {...}, "encoder": {...}}
        options,                         # optional: {"skip_libraries": (...)}
    )

`preprocessing_kind` selects a preprocessing pipeline from
`sklbench.runners.datasets.preprocessing`'s `PREPROCESSORS` - the same
builders `real_datasets.py`'s estimator cases use - via
`sklbench.runners.hptuning._build_pipeline`; there's no runner-side default
or estimator-based special-casing (HGB included), so a dataset with
categorical columns needs one set: `"trees"` below for RandomForest/
ExtraTrees, `"linear"` for Ridge/LogisticRegression, `"hgb"` for
HistGradientBoosting* (caps categorical cardinality to what HGB's native
splitting accepts - see `HGBCategoricalCapper`). Left unset (None), the
estimator sees the raw, unencoded columns - only correct for
already-numeric/already-preprocessed data (e.g. susy below).

`search_space` keys are pipeline step families ("estimator" ->
`estimator__...`, "encoder" -> `preprocessor__encoder__...` - the latter is
`trees_preprocessor`'s ColumnTransformer step name, so it only resolves to
anything when `preprocessing_kind="trees"`), each holding
`{param_name: value}`. A list value is a search dimension (goes to
`hptuning.param_distributions`); anything else is a fixed param (only
supported for "estimator" - `encoder` has no fixed-param case below, so
it's not implemented).
"""

import numpy as np
import math

from joblib import cpu_count
from _implementations import implementations_for_pixi_env

from sklbench.config import Algorithm, Data, HPTuning, HPTuningCase

BENCH = {"n_runs": 3}

REAL_DATASET_CASES = [
    # Trees:
    (
        ("kddcup09_churn", None, "trees"),
        "RandomForestClassifier",
        {   # search space:
            "encoder": {
                "min_frequency": [5, 20, 100],
            },
            "estimator": {
                "n_jobs": -1,
                "n_estimators": [100, 200, 300],
                "max_features": [0.3, "sqrt"],
                "min_samples_split": [2, 5, 20, 100],
                "min_impurity_decrease": [3e-5, 1e-5, 1e-6]
            }
        }
    ),
    (
        ("susy", 300_000),
        "RandomForestClassifier",
        {   # search space:
            "estimator": {
                "n_jobs": -1,
                "n_estimators": 50,
                "min_samples_split": [50, 500, 5000],
                "min_impurity_decrease": [3e-4, 1e-4, 1e-5, 1e-6]
            }
        }
    ),
    (
        ("ames_housing", None, "trees"),
        ["RandomForestRegressor", "ExtraTreesRegressor"],
        {   # search space:
            "encoder": {
                "min_frequency": [1, 5, 20, 100],
            },
            "estimator": {
                "n_jobs": -1,
                "n_estimators": [200, 400, 600],
                "max_features": [1., 0.5, 0.2],
                "max_leaf_nodes": [50, 200, 800],
            }
        }
    ),
    # Linear (only lbfgs for LogisticRegression - see repo notes on
    # class_weight="balanced"/solver choices):
    (
        ("ames_housing", None, "linear"),
        "Ridge",
        {   # search space:
            "estimator": {
                "alpha": list(np.logspace(-3, 3, 13)),
            }
        },
        # Skipped for sklearnex: too slow here (this dataset's "linear"
        # preprocessing refits TargetEncoder's internal KFold on every CV
        # split x candidate) to be worth the matrix size.
        {"skip_libraries": ("sklearnex",)},
    ),
    (
        ("susy", None),
        "Ridge",
        {   # search space:
            "estimator": {
                "alpha": list(np.logspace(-3, 3, 13)),
            }
        },
        {"scoring": "r2"},
    ),
    (
        ("year_prediction_msd", 50_000, "linear"),
        "Ridge",
        {   # search space:
            "estimator": {
                "alpha": list(np.logspace(-3, 3, 13)),
            }
        }
    ),
    (
        ("amazon_employee_access", None, "linear"),
        "LogisticRegression",
        {   # search space:
            "estimator": {
                "solver": "lbfgs",
                "max_iter": 1000,
                "C": list(np.logspace(-3, 3, 13)),
                "fit_intercept": [True, False],
            }
        }
    ),
    (
        ("susy", 500_000),
        "LogisticRegression",
        {   # search space:
            "estimator": {
                "solver": "lbfgs",
                "max_iter": 1000,
                "C": list(np.logspace(-3, 3, 13)),
                "fit_intercept": [True, False],
            }
        }
    ),
    # KMeans (scored on silhouette)
    (
        ("road_network_points", None),
        "KMeans",
        {   # search space:
            "estimator": {
                "n_clusters": [5, 10, 15, 20],
            }
        }
    ),
    (
        ("sift", None),
        "KMeans",
        {   # search space:
            "estimator": {
                "n_clusters": [5, 10, 15, 20],
            }
        }
    ),
    (
        # Downsampled so a many-cluster search still lands near ~5s/fit
        # (nytimes_256 is 290k rows full-size - see real_datasets.py).
        ("nytimes_256", 50_000),
        "KMeans",
        {   # search space:
            "estimator": {
                "n_clusters": [50, 100, 200],
            }
        }
    ),
    # HGB
    (
        ("ames_housing", None, "hgb"),
        "HistGradientBoostingRegressor",
        {   # search space:
            "estimator": {
                "early_stopping": False,
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                "max_leaf_nodes": [7, 15, 31, 63],
                "min_samples_leaf": [5, 10, 20, 50],
                "l2_regularization": [0.0, 0.1, 0.3, 1.0],
                "max_iter": [50, 100, 150],
            }
        },
        {"skip_libraries": ("sklearnex",)},
    ),
    (
        ("kddcup09_churn", None, "hgb"),
        "HistGradientBoostingClassifier",
        {   # search space:
            "estimator": {
                "early_stopping": False,
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                "max_leaf_nodes": [7, 15, 31, 63],
                "min_samples_leaf": [5, 10, 20, 50],
                "l2_regularization": [0.0, 0.1, 0.3, 1.0],
                "max_iter": [50, 100, 150],
            }
        },
        {"skip_libraries": ("sklearnex",)},
    ),
    (
        # No categoricals here, so "hgb" is a no-op passthrough - set
        # anyway for consistency/explicitness with the other two HGB cases.
        ("year_prediction_msd", 30_000, "hgb"),
        "HistGradientBoostingRegressor",
        {   # search space:
            "estimator": {
                "early_stopping": False,
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                "max_leaf_nodes": [7, 15, 31, 63],
                "min_samples_leaf": [5, 10, 20, 50],
                "l2_regularization": [0.0, 0.1, 0.3, 1.0],
                "max_iter": [50, 100, 150],
            }
        },
        {"skip_libraries": ("sklearnex",)},
    ),
]


def _split_search_space(search_space: dict) -> tuple[dict, dict]:
    _FAMILY_PREFIXES = {"estimator": "estimator__", "encoder": "preprocessor__encoder__"}
    estimator_params = {}
    param_distributions = {}
    for family, params in search_space.items():
        prefix = _FAMILY_PREFIXES[family]
        for name, value in params.items():
            if isinstance(value, list):
                param_distributions[f"{prefix}{name}"] = value
            elif family == "estimator":
                estimator_params[name] = value
            else:
                raise ValueError(f"Fixed (non-list) {family!r} param {name!r} isn't supported")
    return estimator_params, param_distributions


def _case(
    dataset: str,
    max_samples: int | None,
    preprocessing_kind: str | None,
    estimator: str,
    search_space: dict,
    implementations: list[dict],
    skip_libraries: tuple[str, ...],
    scoring: str | None,
) -> list[HPTuningCase]:
    estimator_params, param_distributions = _split_search_space(search_space)
    if scoring is None:
        # `hptuning`'s own auto-scoring (see `_default_scoring`) picks
        # roc_auc*/r2 off the *dataset*'s n_classes, regardless of
        # estimator - wrong whenever a case deliberately pairs a regressor
        # with a classification-labeled dataset (e.g. Ridge/susy below),
        # hence the explicit `options={"scoring": ...}` override for those.
        scoring = "silhouette" if estimator == "KMeans" else None
    data = Data(dataset=dataset, preprocessing_kind=preprocessing_kind)
    n_cores = cpu_count(only_physical_cores=True)
    n_cores_list = [round(math.pow(n_cores, v)) for v in [0.5, 0.7, 1]]
    n_iter = n_cores
    if n_cores == 16:
        n_cores_list = [4, 8, 16]
    elif n_cores == 172:
        n_iter = 86
        n_cores_list = [11, 22, 43, 86]

    cases = []
    for implem in implementations:
        if implem["library"] in skip_libraries:
            continue

        for n_jobs in n_cores_list:
            cases.append(HPTuningCase(
                bench=BENCH,
                algorithm=Algorithm(estimator=estimator, estimator_params=estimator_params),
                data=data,
                implementation=implem,
                hptuning=HPTuning(
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    cv_n_splits=3,
                    max_samples=max_samples,
                    n_jobs=n_jobs,
                    scoring=scoring,
                    random_state=3198,
                ),
            ))
    return cases


def generate_cases() -> list[HPTuningCase]:
    # CPU-only, non-array-API implementations
    implementations = [
        implem
        for implem in implementations_for_pixi_env()
        if implem.get("data_library") is None and implem.get("device") in (None, "cpu")
    ]

    cases = []
    for dataset_spec, estimators, search_space, *rest in REAL_DATASET_CASES:
        dataset, max_samples, *preprocessing_kind = dataset_spec
        preprocessing_kind = preprocessing_kind[0] if preprocessing_kind else None
        options = rest[0] if rest else {}
        skip_libraries = options.get("skip_libraries", ())
        scoring = options.get("scoring")

        for estimator in estimators if isinstance(estimators, list) else [estimators]:
            cases.extend(_case(
                dataset, max_samples, preprocessing_kind, estimator, search_space,
                implementations, skip_libraries, scoring,
            ))

    return cases
