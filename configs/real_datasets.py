"""
Real-dataset cases: at least one linear model and one tree-based model per
dataset, run with plain sklearn. Hyperparameters below were picked with a
small RandomizedSearchCV pass per (dataset, model) pair (cv=3,
scoring="roc_auc" for classification / "r2" for regression), not left at
library defaults, so the cases are reasonably realistic rather than
arbitrary.

Most classification/regression datasets also get a HistGradientBoosting
case (`preprocessing_kind="hgb"`: categorical columns are ordinal-encoded
but kept as pandas `category` dtype, so HGB still uses its native
categorical split rather than treating them as plain ordinals). Tuned the
same way as the other models, with `early_stopping=False` fixed during the
search - the goal is a realistic-but-fixed-cost HGB config, since
`configs/hgb_scalability.py` reuses these exact cases (filtered to HGB only) for
its thread-scaling sweep, where a variable iteration count per thread count
would confound the scaling measurement.

Linear cases don't request Nystroem kernel approximation (the default of
`linear_preprocessor` in `sklbench/runners/datasets/preprocessing.py`): it
was measured to hurt held-out performance on every dataset below (e.g. ROC
AUC -0.02 to -0.06), unless the case is specifically exercising the Nystroem
path (`kick`'s second LogisticRegression case below), which opts in via
`preprocessing_kwargs={"nystroem": {}}`.

The clustering cases (`sift`, `nytimes_256`, `fashion_mnist_784`) are dense
embedding datasets from ann-benchmarks.com and only get a single KMeans
case each, run at library-default hyperparameters aside from `n_clusters`:
unlike the linear/tree cases above, the interesting axis here is
n_samples/n_clusters, not per-model hyperparameter tuning.
"""
from typing import Callable, Iterable
from math import floor

from joblib import cpu_count

from sklbench.config.utils import select_logistic_regression_solver

BENCH = {"n_runs": 1, "py_spy_profiling": False}

# KMeans crashes (segfault / heap corruption) on this repo's many-core
# machines when OpenMP spins up as many threads as there are cores, during
# joblib/OpenMP nested parallelism - see
# https://github.com/OpenMathLib/OpenBLAS/issues/5958. Cap it below that.
KMEANS_BENCH = {"env": 
    {"OMP_NUM_THREADS": "128"} if cpu_count(only_physical_cores=True) > 128
    else {}
}

N_JOBS = floor(0.9 * cpu_count(only_physical_cores=True))
# RF/ET n_estimators below are `max(<tuned floor>, N_JOBS * 3)`: use at least
# as many trees as were tuned on a typical dev machine, but scale up on
# many-core machines so the forest actually uses the available parallelism.

DEFAULT_PREPROCESSING_KIND = {
    "Ridge": "linear",
    "LogisticRegression": "linear",
    "RandomForestRegressor": "trees",
    "RandomForestClassifier": "trees",
    "ExtraTreesClassifier": "trees",
    "ExtraTreesRegressor": "trees",
    "HistGradientBoostingClassifier": "hgb",
    "HistGradientBoostingRegressor": "hgb",
    "KMeans": None,
}

# TODO? use Ridge on some clf datasets
# or ask sklearnex to support RidgeClassifier?


TIERS = ("test", "fast", "normal")

REAL_DATASET_CASE_FUNCS: list[Callable] = []


def real_case_dataset(dataset, task, tier):

    def decorator(f: Callable[[dict], Iterable[dict]]):

        def decorated(implem: dict | None = None, bench: dict | None = None):
            if implem is None:
                implem = {"library": "sklearn"}
            bench = bench or BENCH
            for entry in f(implem):
                entry_tier = entry.pop("tier", tier)
                split_kwargs = entry.pop("split_kwargs", None) or {}
                preprocessing_kwargs = entry.pop("preprocessing_kwargs", None) or {}
                preprocessing_kind = entry.pop(
                    "preprocessing_kind",
                    DEFAULT_PREPROCESSING_KIND[entry["estimator"]],
                )
                entry_bench = entry.pop("bench", None)
                case_bench = {**bench, **entry_bench} if entry_bench else bench
                yield {
                    "bench": case_bench,
                    "implementation": implem,
                    "metadata": {"task": task, "tier": entry_tier},
                    "algorithm": {**entry},
                    "data": {
                        "dataset": dataset,
                        "split_kwargs": split_kwargs,
                        "preprocessing_kind": preprocessing_kind,
                        "preprocessing_kwargs": preprocessing_kwargs,
                    },
                }

        REAL_DATASET_CASE_FUNCS.append(decorated)
        return decorated

    return decorator


@real_case_dataset("ames_housing", "regression", "fast")
def ames_housing(implem: dict):
    # Ridge(alpha=1.0), linear/no-nystroem: test R2 0.89
    yield {
        "estimator": "Ridge",
        "estimator_params": {"alpha": 1.0},
    }
    # RandomForestRegressor, trees/ordinal: test R2 0.89
    yield {
        "estimator": "RandomForestRegressor",
        "estimator_params": {
            "n_estimators": max(300, N_JOBS * 6),
            "max_depth": 20,
            "max_features": 0.5,
            "min_samples_leaf": 1,
            "n_jobs": N_JOBS,
        },
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingRegressor, hgb/native-categorical: test R2 0.898
        yield {
            "estimator": "HistGradientBoostingRegressor",
            "estimator_params": {
                "learning_rate": 0.17,
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 10,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("kddcup09_churn", "classification", "normal")
def kddcup(implem: dict):
    # Severe imbalance (7.3% churn).
    solver = select_logistic_regression_solver(implem, ["newton-cholesky", "lbfgs"])
    # LogisticRegression, linear/no-nystroem: ROC AUC 0.698
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 5.5,
            "solver": solver,
            "max_iter": 1000,
        },
    }
    # RandomForestClassifier, trees/ordinal: ROC AUC 0.713
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": N_JOBS * 6,
            "max_depth": 20,
            "max_features": 0.3,
            "min_samples_leaf": 10,
            "n_jobs": N_JOBS,
        },
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingClassifier, hgb/native-categorical: ROC AUC 0.690
        yield {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "learning_rate": 0.024,
                "max_iter": 150,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 50,
                "l2_regularization": 0.3,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("amazon_employee_access", "classification", "fast")
def amazon_employee_access(implem: dict):
    # LogisticRegression, linear/no-nystroem: ROC AUC 0.843
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 1,
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    }
    # RandomForestClassifier, trees/ordinal: ROC AUC 0.855
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": max(200, N_JOBS * 4),
            "max_features": 0.3,
            "min_samples_leaf": 2,
            "n_jobs": N_JOBS,
        },
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingClassifier, hgb/native-categorical: ROC AUC 0.857
        yield {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "learning_rate": 0.021,
                "max_iter": 150,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("kick", "classification", "fast")
def kick(implem: dict):
    # LogisticRegression w/ Nystroem (poly deg-2, 300 components): ROC AUC 0.763
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "solver": "lbfgs",
            "max_iter": 2000,
            "C": 24.0,
        },
        "preprocessing_kwargs": {"nystroem": {}},
    }
    # ExtraTreesClassifier, trees/ordinal: ROC AUC 0.760
    yield {
        "estimator": "ExtraTreesClassifier",
        "estimator_params": {
            "n_estimators": max(200, N_JOBS * 4),
            "max_depth": 30,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "n_jobs": N_JOBS,
        },
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingClassifier, hgb/native-categorical: ROC AUC 0.764
        yield {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "learning_rate": 0.045,
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 5,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("covtype", "classification", "normal")
def covtype(implem: dict):
    # Environmental/forestry vertical. 581012 x 12: 10 continuous numeric
    # (elevation, slope, hillshade, distances, ...) plus 2 categorical
    # features (`Wilderness_Area` 4 cats, `Soil_Type` 40 cats)
    # reconstructed by `undo_one_hot` from their one-hot-encoded source
    # columns, so this exercises real categorical preprocessing rather
    # than scaling 44 already-binary columns. 7-class classification,
    # severely imbalanced (smallest class is 0.47% of rows).

    # LogisticRegression, linear/poly-nystroem (100 components): ROC AUC
    # (OVR, weighted) 0.881, ~48s/fit
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 10.0,
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "preprocessing_kwargs": {"nystroem": {"n_components": 100}},
    }
    # RandomForestClassifier, trees/ordinal: ROC AUC (OVR, weighted) 0.997
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": max(100, N_JOBS * 3),
            "n_jobs": N_JOBS,
        },
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingClassifier, hgb/native-categorical: ROC AUC
        # (OVR, weighted) 0.992
        yield {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "learning_rate": 0.070,
                "max_iter": 200,
                "max_leaf_nodes": 127,
                "min_samples_leaf": 5,
                "l2_regularization": 0.05,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("susy", "classification", "normal")
def susy(implem: dict):
    # This dataset's features are already standardized by the source
    # (means ~0/1, stds all in [0.2, 1.0]), so no scaling/preprocessing
    # is needed for the linear case either.
    solver = select_logistic_regression_solver(implem, ["newton-cholesky", "lbfgs"])
    # LogisticRegression, no preprocessing: ROC AUC 0.859
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "max_iter": 200,
            "solver": solver,
        },
        "preprocessing_kind": None,
    }
    # ExtraTreesClassifier, no preprocessing: ROC AUC 0.869. ~20-30s/fit at
    # 4.5M rows (RandomForest doesn't scale as gracefully as
    # histogram-based boosting).
    yield {
        "estimator": "ExtraTreesClassifier",
        "estimator_params": {
            "n_estimators": max(50, N_JOBS * 2),
            "min_impurity_decrease": 3e-6,
            "max_depth": 16,
            "n_jobs": N_JOBS,
        },
        "preprocessing_kind": None,
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingClassifier, hgb (no categorical columns, so
        # this is a no-op passthrough - kept for consistency with the other
        # cases below): ROC AUC 0.878
        yield {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "learning_rate": 0.070,
                "max_iter": 200,
                "max_leaf_nodes": 127,
                "min_samples_leaf": 5,
                "l2_regularization": 0.05,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("year_prediction_msd", "regression", "normal")
def year_prediction_msd(implem: dict):
    # Music/audio vertical. 515345 x 90, all numeric, regression
    # (predict a song's release year from timbre features). Inherently
    # hard task - R2 ~0.2-0.3 matches published baselines for this
    # dataset, not an undertuned model.

    # Ridge(alpha=1.0), linear/no-nystroem: test R2 0.266. Splines
    # place knots per-column, so they don't need prior scaling
    # despite the raw features spanning wildly different scales
    # (stds range from ~6 to ~1749 across columns).
    yield {
        "estimator": "Ridge",
        "estimator_params": {"alpha": 1.0},
    }
    # RandomForestRegressor, no preprocessing: test R2 0.242, ~75s/fit
    # (kept small - 100 trees/depth 16 took 441s for barely better R2).
    # time_limit bumped to 600s: at the default (300s), only 2/7 n_runs
    # repeats were completing before timing out.
    yield {
        "estimator": "RandomForestRegressor",
        "estimator_params": {
            "n_estimators": max(30, N_JOBS * 2),
            "max_depth": 10,
            "n_jobs": N_JOBS,
        },
        "preprocessing_kind": None,
        "bench": {"time_limit": 600},
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingRegressor, hgb (no categorical columns): test
        # R2 0.314 - better than both Ridge and RandomForest above, though
        # still firmly in "inherently hard task" territory like they are.
        yield {
            "estimator": "HistGradientBoostingRegressor",
            "estimator_params": {
                "learning_rate": 0.070,
                "max_iter": 200,
                "max_leaf_nodes": 127,
                "min_samples_leaf": 5,
                "l2_regularization": 0.05,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("fraud", "classification", "normal")
def fraud(implem: dict):
    # Finance/fraud vertical. 284807 x 30 (PCA-anonymized transaction
    # features), all numeric. Extreme imbalance (0.17% fraud) - much
    # more severe than kddcup09_churn's 7.3%. The loader flags its V1-V28
    # PCA columns as linear-preprocessing passthrough (already
    # well-conditioned, not spline-expanded) and spline-encodes
    # Time_of_day/Amount with 20 knots instead of the default 10 - see
    # `load_fraud`'s docstring and `data_desc["preprocessing_defaults"]`.
    solver = select_logistic_regression_solver(implem, ["newton-cholesky", "lbfgs"])
    # LogisticRegression, linear/no-nystroem: ROC AUC 0.984
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "max_iter": 1000,
            "solver": solver,
        },
    }
    # RandomForestClassifier, no preprocessing: ROC AUC 0.949
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": max(200, N_JOBS * 4),
            "n_jobs": N_JOBS,
        },
        "preprocessing_kind": None,
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingClassifier, hgb (no categorical columns): ROC
        # AUC 0.982
        yield {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "learning_rate": 0.024,
                "max_iter": 150,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 50,
                "l2_regularization": 0.3,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("medical_charges_nominal", "regression", "normal")
def medical_charges_nominal(implem: dict):
    # Healthcare vertical. 163065 x 11, regression (predict average
    # hospital payment per diagnosis-related group/provider). Standard
    # benchmark for very-high-cardinality categorical encoding: several
    # provider-identity columns have ~2000-3300 categories each.

    # Ridge(alpha=1.0), linear/no-nystroem: test R2 0.982
    yield {
        "estimator": "Ridge",
        "estimator_params": {"alpha": 1.0},
    }
    # RandomForestRegressor, trees/ordinal: test R2 0.982
    yield {
        "estimator": "RandomForestRegressor",
        "estimator_params": {
            "n_estimators": max(300, N_JOBS * 6),
            "max_depth": 20,
            "max_features": 0.5,
            "n_jobs": N_JOBS,
        },
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingRegressor, hgb/native-categorical: test R2 0.978
        yield {
            "estimator": "HistGradientBoostingRegressor",
            "estimator_params": {
                "learning_rate": 0.057,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("bank_marketing", "classification", "fast")
def bank_marketing(implem: dict):
    # Banking/finance vertical (distinct from `fraud`'s transaction-fraud
    # angle: this is telemarketing conversion). 45211 x 16, 9 low-
    # cardinality categoricals (2-12 categories) + numeric. Moderate
    # imbalance (~11.7% positive).

    # LogisticRegression, linear/no-nystroem: ROC AUC 0.913
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 1.57,
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    }
    # RandomForestClassifier, trees/ordinal: ROC AUC 0.924
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": max(300, N_JOBS * 6),
            "max_depth": 20,
            "max_features": 0.3,
            "min_samples_leaf": 2,
            "n_jobs": N_JOBS,
        },
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingClassifier, hgb/native-categorical: ROC AUC 0.934
        yield {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "learning_rate": 0.057,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("california_housing", "regression", "fast")
def california_housing(implem: dict):
    # 1990 US census vertical (distinct from `ames_housing`'s per-sale-listing
    # angle: this is aggregated per census block group). 20640 x 8, all
    # numeric, no missing values - a clean, small counterpart to the
    # higher-dimensional/messier regression datasets above.

    # Ridge(alpha=0.3), linear/no-nystroem: test R2 0.658
    yield {
        "estimator": "Ridge",
        "estimator_params": {"alpha": 0.3},
    }
    # RandomForestRegressor, no preprocessing (all-numeric): test R2 0.817
    yield {
        "estimator": "RandomForestRegressor",
        "estimator_params": {
            "n_estimators": max(300, N_JOBS * 6),
            "max_features": 0.3,
            "min_samples_leaf": 1,
            "n_jobs": N_JOBS,
        },
        "preprocessing_kind": None,
    }
    if implem["library"] == "sklearn":
        # HistGradientBoostingRegressor, hgb (no categorical columns): test
        # R2 0.852
        yield {
            "estimator": "HistGradientBoostingRegressor",
            "estimator_params": {
                "learning_rate": 0.1,
                "max_iter": 150,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 20,
                "max_bins": 255,
                "early_stopping": False,
            },
        }


@real_case_dataset("susy", "regression", "normal")
def susy_regression(implem: dict):
    # Reuses `susy`'s classification data (see the `susy` case above) but
    # regresses on the 0/1 signal/background label directly instead of
    # classifying it - meaningful here specifically because susy's target is
    # close to balanced (45.7% positive), unlike the other classification
    # datasets in this file. Only Ridge is included: RandomForestRegressor/
    # HistGradientBoostingRegressor on the same label wouldn't add anything
    # over the tree-based classification cases already covering this
    # dataset above.

    # Ridge(alpha=10.0), no preprocessing (already standardized): test R2
    # 0.276 - unsurprisingly low compared to the classification case's ROC
    # AUC 0.859, since Ridge is fitting a linear regressor to a hard
    # nonlinear boundary encoded as a 0/1 target.
    yield {
        "estimator": "Ridge",
        "estimator_params": {"alpha": 10.0},
        "preprocessing_kind": None,
    }


@real_case_dataset("sift", "clustering", "normal")
def sift(implem: dict):
    # ANN-benchmarks vertical: SIFT image descriptors, 128-dim, Euclidean
    # distance native (no L2-normalization needed, unlike the angular
    # datasets below). 1,000,000 train vectors by default.

    # KMeans, full dataset, few clusters: ~3s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 10},
        "tier": "fast",
        "bench": KMEANS_BENCH,
    }
    # KMeans, full dataset, many clusters: ~29s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 100},
        "bench": KMEANS_BENCH,
    }
    # n_samples/n_clusters ~= 100 (10000/100).
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 1000},
        "split_kwargs": {"train_size": 100000, "test_size": 1000},
        "bench": KMEANS_BENCH,
    }


@real_case_dataset("nytimes_256", "clustering", "fast")
def nytimes_256(implem: dict):
    # ANN-benchmarks vertical: NYTimes bag-of-words document embeddings,
    # 256-dim. The loader L2-normalizes so Euclidean distance (and thus
    # k-means) matches the dataset's native cosine/angular distance.
    # 290,000 train vectors by default.

    # KMeans, full dataset, few clusters: ~2s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 10},
        "bench": KMEANS_BENCH,
    }
    # KMeans, full dataset, many clusters: ~15s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 100},
        "tier": "normal",
        "bench": KMEANS_BENCH,
    }
    # Small subsample via split_kwargs: a near-instant sanity-check case for
    # validating the config wiring, not a realistic workload.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 5},
        "split_kwargs": {"train_size": 2000, "test_size": 200},
        "tier": "test",
        "bench": KMEANS_BENCH,
    }


@real_case_dataset("fashion_mnist_784", "clustering", "normal")
def fashion_mnist_784(implem: dict):
    # ANN-benchmarks vertical: Fashion-MNIST images flattened to 784-dim
    # pixel vectors, Euclidean distance native. 60,000 train vectors by
    # default. KMeans, full dataset, many clusters: ~7.5s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 100},
        "bench": KMEANS_BENCH,
    }


@real_case_dataset("road_network_points", "clustering", "fast")
def road_network_points(implem: dict):
    # Genuine low-dimensional spatial vertical (contrast with the dense
    # embedding datasets above): 3D Road Network GPS points, 434874 x 3
    # (longitude/latitude/altitude), no target. Being only 3-dim, k-means is
    # cheap per cluster - reaching a "normal"-tier fit time needs far more
    # clusters here than for the ~100-1000 dim embedding datasets.

    # KMeans, full dataset, few clusters: ~0.2s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 10},
        "bench": KMEANS_BENCH,
    }
    # KMeans, full dataset, many clusters: ~14s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 1000},
        "tier": "normal",
        "bench": KMEANS_BENCH,
    }


def generate_cases(implem: dict | None = None, max_tier: str = "normal") -> list[dict]:
    max_tier_index = TIERS.index(max_tier)
    cases = []
    for case_func in REAL_DATASET_CASE_FUNCS:
        for case in case_func(implem):
            tier = case["metadata"]["tier"]
            if tier == "test" and max_tier != "test":
                continue
            if TIERS.index(tier) > max_tier_index:
                continue
            cases.append(case)
    return cases
