"""
RandomizedSearchCV cases over a short, hand-picked list of real datasets:

Each `REAL_DATASET_CASES` entry is:

    (
        (dataset, max_samples),          # or (dataset, max_samples, preprocessing_kind)
        estimator,                       # str, or list[str] to reuse one search space
        search_space,                    # {"estimator": {...}, "encoder": {...}}
        options,                         # optional: {"skip_libraries": (...)}
    )

"""

import numpy as np
import math

from joblib import cpu_count
from _utils.implementations import implementations_for_pixi_env

from sklbench.config import Algorithm, Data, HPTuning, HPTuningCase

BENCH = {"n_runs": 3}
N_ESTIMATORS = 2 * cpu_count() if cpu_count() <= 32 else cpu_count()

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
                "n_estimators": N_ESTIMATORS,
                "max_features": "sqrt",
                "min_samples_split": [5, 20, 100],
                "min_impurity_decrease": [3e-5, 1e-5, 1e-6]
            }
        },
        # sklearnex's oneDAL RandomForestClassifier doesn't support missing values:
        {"preprocessing_kwargs_by_library": {"sklearnex": {"remove_nans": True}}},
    ),
    (
        ("susy", 300_000),
        "RandomForestClassifier",
        {   # search space:
            "estimator": {
                "n_jobs": -1,
                "n_estimators": N_ESTIMATORS,
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
                "n_estimators": N_ESTIMATORS * 2,
                "max_features": [1., 0.5, 0.2],
                "max_leaf_nodes": [50, 200, 800],
            }
        },
        # ames_housing has real missing values too (see loader docstring);
        # sklearnex's oneDAL RF/ET don't support them.
        {"preprocessing_kwargs_by_library": {"sklearnex": {"remove_nans": True}}},
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


KMEANS_CASES = [
    # KMeans (scored on silhouette); SKIPPED FOR NOW;
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
        # (nytimes_256 is 290k rows full-size - see _real_datasets.py).
        ("nytimes_256", 50_000),
        "KMeans",
        {   # search space:
            "estimator": {
                "n_clusters": [50, 100, 200],
            }
        }
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


def get_n_iter_and_n_jobs_list(estimator: str):
    TREES = [
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesRegressor", "ExtraTreesClassifier"
    ]
    is_tree = estimator in TREES
    n_cores = cpu_count(only_physical_cores=True)
    n_iter = n_cores * 2
    # default:
    n_jobs_list = [round(math.pow(n_cores, v)) for v in [0.5, 0.7, 1]]
    if is_tree:
        n_jobs_list = [1, 2, *n_jobs_list]
    n_jobs_list = sorted(set([min(n_jobs, n_cores) for n_jobs in n_jobs_list]))

    if n_cores == 16:
        if is_tree:
            n_jobs_list = [1, 2, 4, 8, 16]
        else:
            n_jobs_list = [4, 8, 16]

    elif n_cores == 172:
        n_iter = n_cores
        if is_tree:
            n_jobs_list = [1, 2, 5, 11, 22, 43, 86]
        else:
            n_jobs_list = [11, 22, 43, 86]

    return n_iter, n_jobs_list


def _case(
    dataset: str,
    max_samples: int | None,
    preprocessing_kind: str | None,
    estimator: str,
    search_space: dict,
    implementations: list[dict],
    skip_libraries: tuple[str, ...],
    scoring: str | None,
    preprocessing_kwargs_by_library: dict[str, dict] | None,
) -> list[HPTuningCase]:
    estimator_params, param_distributions = _split_search_space(search_space)
    if scoring is None:
        # `hptuning`'s own auto-scoring (see `_default_scoring`) picks
        # roc_auc*/r2 off the *dataset*'s n_classes, regardless of
        # estimator - wrong whenever a case deliberately pairs a regressor
        # with a classification-labeled dataset (e.g. Ridge/susy below),
        # hence the explicit `options={"scoring": ...}` override for those.
        scoring = "silhouette" if estimator == "KMeans" else None
    preprocessing_kwargs_by_library = preprocessing_kwargs_by_library or {}

    cases = []
    for implem in implementations:
        if implem["library"] in skip_libraries:
            continue

        data = Data(
            dataset=dataset,
            preprocessing_kind=preprocessing_kind,
            preprocessing_kwargs=preprocessing_kwargs_by_library.get(implem["library"], {}),
        )

        n_iter, n_jobs_list = get_n_iter_and_n_jobs_list(estimator)

        for n_jobs in n_jobs_list:
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
        preprocessing_kwargs_by_library = options.get("preprocessing_kwargs_by_library")

        for estimator in estimators if isinstance(estimators, list) else [estimators]:
            cases.extend(_case(
                dataset, max_samples, preprocessing_kind, estimator, search_space,
                implementations, skip_libraries, scoring, preprocessing_kwargs_by_library,
            ))

    return cases
