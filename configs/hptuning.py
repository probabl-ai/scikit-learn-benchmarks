"""
Outer-parallelism benchmark of a hyper-parameter search.

Each case sweeps the search's outer `n_jobs` and fits `K` fixed candidates,
each on `n_jobs` random 80/20 train/test splits (`K * n_jobs` fits). Every
outer worker has `K` fits to do at every point of the sweep, and each fit is
the same on every machine (fixed data size and `n_estimators`), so the time
per fit compares `n_jobs` points and machines.

Each `REAL_DATASET_CASES` entry is:

    (
        (dataset, max_samples),          # or (dataset, max_samples, preprocessing_kind)
        estimator,                       # str, or list[str] to reuse one search space
        search_space,                    # {"estimator": {...}, "encoder": {...}}
        options,                         # optional: {"skip_libraries": (...)}
    )

List values in `search_space` form the grid that `_candidates` draws the `K`
candidates from, other values are fixed estimator params.
"""

import itertools

import numpy as np
import math

from joblib import cpu_count
from _utils.implementations import implementations_for_pixi_env

from sklbench.config import Algorithm, Data, HPTuning, HPTuningCase

BENCH = {"n_runs": 3}
K = 8
# Data sizes keep a single-threaded fit under ~3s, so each worker has at
# most ~K * 3s of work.
N_CORES = cpu_count(only_physical_cores=True)
# Row counts of the datasets used at full size (`max_samples=None`), for
# sklearnex's tree `max_bins`.
DATASET_N_SAMPLES = {"ames_housing": 1_460}
TREES = [
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesRegressor", "ExtraTreesClassifier"
]
N_ESTIMATORS = 100

REAL_DATASET_CASES = [
    # Trees:
    (
        ("kddcup09_churn", 20_000, "trees"),
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
        ("susy", 20_000),
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
                "alpha": list(np.logspace(-3, 3, K)),
            }
        },
        # Skipped for sklearnex: too slow here (this dataset's "linear"
        # preprocessing refits TargetEncoder's internal KFold on every CV
        # split x candidate) to be worth the matrix size.
        # TODO: fix comment and maybe stop skipping sklearnex. Until 0e5acafa,
        # TargetEncoder treated this integer regression target as ~600
        # classes, giving ~25k features instead of ~660, which likely caused
        # most of the slowness.
        {"skip_libraries": ("sklearnex",)},
    ),
    (
        ("susy", None),
        "Ridge",
        {   # search space:
            "estimator": {
                "alpha": list(np.logspace(-3, 3, K)),
            }
        },
        {"scoring": "r2"},
    ),
    (
        ("year_prediction_msd", 50_000, "linear"),
        "Ridge",
        {   # search space:
            "estimator": {
                "alpha": list(np.logspace(-3, 3, K)),
            }
        },
    ),
    (
        ("amazon_employee_access", None, "linear"),
        "LogisticRegression",
        {   # search space:
            "estimator": {
                "solver": "lbfgs",
                "max_iter": 1000,
                "C": list(np.logspace(-3, 3, K // 2)),
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
                "C": list(np.logspace(-3, 3, K // 2)),
                "fit_intercept": [True, False],
            }
        },
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
        ("kddcup09_churn", 20_000, "hgb"),
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
    grid = {}
    for family, params in search_space.items():
        prefix = _FAMILY_PREFIXES[family]
        for name, value in params.items():
            if isinstance(value, list):
                grid[f"{prefix}{name}"] = value
            elif family == "estimator":
                estimator_params[name] = value
            else:
                raise ValueError(f"Fixed (non-list) {family!r} param {name!r} isn't supported")
    return estimator_params, grid


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _candidates(grid: dict, k: int = K, seed: int = 0) -> list[dict]:
    """`k` distinct points of `grid`, the same on every machine (legacy
    `RandomState` streams don't change across numpy versions)."""
    names = sorted(grid)
    points = list(itertools.product(*(grid[name] for name in names)))
    if len(points) < k:
        raise ValueError(f"The grid {grid} has {len(points)} points, fewer than {k}")
    indices = np.random.RandomState(seed).choice(len(points), size=k, replace=False)
    return [
        {name: _python_scalar(value) for name, value in zip(names, points[index])}
        for index in sorted(indices)
    ]


def get_n_jobs_list(estimator: str, strong_scaling=False) -> list[int]:
    if strong_scaling:
        # A set since `N_CORES // 4` and `N_CORES // 2` can both be 1 on small CI machines.
        n_jobs_list = sorted({max(N_CORES // 4, 1), max(N_CORES // 2, 1), N_CORES})
        if N_CORES == 172:
            n_jobs_list = [22, 43, 86, 172]
        return n_jobs_list
    else:
        # 1, 2, 4, ..., n_cores
        n_jobs_list = [*[2**i for i in range((N_CORES - 1).bit_length())], N_CORES]
        if N_CORES == 172:
            n_jobs_list = [1, 2, 5, 11, 22, 43, 86, 172]
        return n_jobs_list


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
    strong_scaling: bool,
) -> list[HPTuningCase]:
    estimator_params, grid = _split_search_space(search_space)
    candidates = _candidates(grid)
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

        implem_estimator_params = estimator_params
        if implem["library"] == "sklearnex" and estimator in TREES:
            # Exact (unbinned) splits, like sklearn's
            n_samples = max_samples if max_samples is not None else DATASET_N_SAMPLES[dataset]
            implem_estimator_params = {**estimator_params, "max_bins": n_samples}

        for n_jobs in get_n_jobs_list(estimator, strong_scaling):
            cases.append(HPTuningCase(
                bench=BENCH,
                algorithm=Algorithm(estimator=estimator, estimator_params=implem_estimator_params),
                data=data,
                implementation=implem,
                hptuning=HPTuning(
                    candidates=candidates,
                    n_splits=N_CORES if strong_scaling else n_jobs,
                    max_samples=max_samples,
                    n_jobs=n_jobs,
                    scoring=scoring,
                    random_state=3198,
                ),
            ))
    return cases


def generate_cases(strong_scaling: bool = False) -> list[HPTuningCase]:
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
                strong_scaling,
            ))

    return cases
