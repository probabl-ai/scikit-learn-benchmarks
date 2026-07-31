"""
Real-dataset cases: at least one linear model and one tree-based model per
dataset, run with plain sklearn. Hyperparameters below were picked with a
small RandomizedSearchCV pass per (dataset, model) pair (cv=3,
scoring="roc_auc" for classification / "r2" for regression), not left at
library defaults, so the cases are reasonably realistic rather than
arbitrary.

Linear cases don't request Nystroem kernel approximation (the default of
`linear_preprocessor` in `sklbench/runners/datasets/preprocessing.py`): it
was measured to hurt held-out performance on every dataset below (e.g. ROC
AUC -0.02 to -0.06), unless the case is specifically exercising the Nystroem
path (`kick`'s second LogisticRegression case below), which opts in via
`preprocessing_kwargs={"nystroem": {}}`.

The clustering cases (`sift`, `nytimes_256`, `fashion_mnist_784`) are dense
embedding datasets from ann-benchmarks.com and only get a single KMeans
case each, run at library-default hyperparameters aside from `n_clusters`
and `random_state`: unlike the linear/tree cases above, the interesting
axis here is n_samples/n_clusters, not per-model hyperparameter tuning.
"""
from typing import Callable, Iterable

from joblib import cpu_count

from sklbench.config.utils import select_logistic_regression_solver

BENCH = {"n_runs": 1, "py_spy_profiling": False}

N_JOBS = cpu_count(only_physical_cores=True)

DEFAULT_PREPROCESSING_KIND = {
    "Ridge": "linear",
    "LogisticRegression": "linear",
    "RandomForestRegressor": "trees",
    "RandomForestClassifier": "trees",
    "ExtraTreesClassifier": "trees",
    "ExtraTreesRegressor": "trees",
    "KMeans": None,
}


TIERS = ("test", "fast", "normal", "slow")

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
                yield {
                    "bench": bench,
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


@real_case_dataset("ames_housing", "regression", "test")
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
            "n_estimators": 300,
            "max_depth": 20,
            "max_features": 0.5,
            "min_samples_leaf": 1,
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
    }


@real_case_dataset("kddcup09_churn", "classification", "normal")
def kddcup(implem: dict):
    # Severe imbalance (7.3% churn) makes the best-by-ROC-AUC model for
    # both estimators collapse to predicting the majority class
    # (balanced accuracy ~0.50, i.e. useless for churn detection), so
    # class_weight="balanced"/"balanced_subsample" is used instead - a
    # small ROC AUC trade for a model that actually discriminates churn.
    solver = select_logistic_regression_solver(implem, ["newton-cholesky", "lbfgs"])
    # LogisticRegression, linear/no-nystroem: test acc 0.681, bal.acc 0.640, ROC AUC 0.698
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 5.5,
            "class_weight": "balanced",
            "solver": solver,
            "max_iter": 1000,
        },
    }
    # RandomForestClassifier, trees/ordinal: test acc 0.909, bal.acc 0.558, ROC AUC 0.713
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": N_JOBS * 6,
            "max_depth": 20,
            "max_features": 0.3,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
    }


@real_case_dataset("amazon_employee_access", "classification", "fast")
def amazon_employee_access(implem: dict):
    # LogisticRegression, linear/no-nystroem: test acc 0.826, bal.acc 0.794, ROC AUC 0.843
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 1,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    }
    # RandomForestClassifier, trees/ordinal: test acc 0.950, bal.acc 0.606, ROC AUC 0.855
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": 200,
            "max_features": 0.3,
            "min_samples_leaf": 2,
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
    }


@real_case_dataset("kick", "classification", "fast")
def kick(implem: dict):
    # LogisticRegression w/ Nystroem (poly deg-2, 300 components):
    # test acc 0.737, bal.acc 0.697, ROC AUC 0.768 - close to the
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "solver": "lbfgs",
            "max_iter": 2000,
            "C": 24.0,
            "class_weight": "balanced",
        },
        "preprocessing_kwargs": {"nystroem": {}},
    }
    # ExtraTreesClassifier, trees/ordinal: test acc 0.809, bal.acc 0.684, ROC AUC 0.760
    yield {
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
    }


@real_case_dataset("covtype", "classification", "normal")
def covtype(implem: dict):
    # Environmental/forestry vertical. 581012 x 12: 10 continuous numeric
    # (elevation, slope, hillshade, distances, ...) plus 2 categorical
    # features (`Wilderness_Area` 4 cats, `Soil_Type` 40 cats)
    # reconstructed by `undo_one_hot` from their one-hot-encoded source
    # columns, so this exercises real categorical preprocessing rather
    # than scaling 44 already-binary columns. 7-class classification,
    # severely imbalanced (smallest class is 0.47% of rows), so
    # class_weight="balanced" is used for the linear case; RandomForest
    # handles the imbalance fine on its own.

    # LogisticRegression, linear/poly-nystroem (100 components): test acc
    # 0.629, bal.acc 0.752, ~48s/fit
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 10.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "preprocessing_kwargs": {"nystroem": {"n_components": 100}},
    }
    # RandomForestClassifier, trees/ordinal: test acc 0.963, bal.acc 0.921
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": 100,
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
    }


@real_case_dataset("susy", "classification", "normal")
def susy(implem: dict):
    # This dataset's features are already standardized by the source
    # (means ~0/1, stds all in [0.2, 1.0]), so no scaling/preprocessing
    # is needed for the linear case either.
    solver = select_logistic_regression_solver(implem, ["newton-cholesky", "lbfgs"])
    # LogisticRegression, no preprocessing: test acc 0.789, bal.acc 0.780, ROC AUC 0.859
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "max_iter": 200,
            "solver": solver,
        },
        "preprocessing_kind": None,
    }
    # ExtraTreesClassifier, no preprocessing: test acc 0.798, bal.acc 0.790,
    # ROC AUC 0.869. ~20-30s/fit at 4.5M rows (RandomForest doesn't scale as
    # gracefully as histogram-based boosting).
    yield {
        "estimator": "ExtraTreesClassifier",
        "estimator_params": {
            "n_estimators": 50,
            "min_impurity_decrease": 3e-6,
            "max_depth": 16,
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
        "preprocessing_kind": None,
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
    # (kept small - 100 trees/depth 16 took 441s for barely better R2)
    yield {
        "estimator": "RandomForestRegressor",
        "estimator_params": {
            "n_estimators": 30,
            "max_depth": 10,
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
        "preprocessing_kind": None,
    }


@real_case_dataset("fraud", "classification", "normal")
def fraud(implem: dict):
    # Finance/fraud vertical. 284807 x 30 (PCA-anonymized transaction
    # features), all numeric. Extreme imbalance (0.17% fraud) - much
    # more severe than kddcup09_churn's 7.3%, so class_weight="balanced"
    # matters even more here. The loader flags its V1-V28 PCA columns as
    # linear-preprocessing passthrough (already well-conditioned, not
    # spline-expanded) and spline-encodes Time_of_day/Amount with 20
    # knots instead of the default 10 - see `load_fraud`'s docstring and
    # `data_desc["preprocessing_defaults"]`.
    solver = select_logistic_regression_solver(implem, ["newton-cholesky", "lbfgs"])
    # LogisticRegression, linear/no-nystroem: test acc 0.946, bal.acc 0.939, ROC AUC 0.984
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": solver,
        },
    }
    # RandomForestClassifier, no preprocessing: test acc 1.000 (rounded),
    # bal.acc 0.860, ROC AUC 0.969
    yield {
        "estimator": "RandomForestClassifier",
        "estimator_params": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
        "preprocessing_kind": None,
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
            "n_estimators": 300,
            "max_depth": 20,
            "max_features": 0.5,
            "random_state": 0,
            "n_jobs": N_JOBS,
        },
    }


@real_case_dataset("bank_marketing", "classification", "fast")
def bank_marketing(implem: dict):
    # Banking/finance vertical (distinct from `fraud`'s transaction-fraud
    # angle: this is telemarketing conversion). 45211 x 16, 9 low-
    # cardinality categoricals (2-12 categories) + numeric. Moderate
    # imbalance (~11.7% positive).

    # LogisticRegression, linear/no-nystroem: test acc 0.838, bal.acc 0.843, ROC AUC 0.913
    yield {
        "estimator": "LogisticRegression",
        "estimator_params": {
            "C": 1.57,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    }
    # RandomForestClassifier, trees/ordinal: test acc 0.900, bal.acc 0.776, ROC AUC 0.924
    yield {
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
    }


@real_case_dataset("sift", "clustering", "normal")
def sift(implem: dict):
    # ANN-benchmarks vertical: SIFT image descriptors, 128-dim, Euclidean
    # distance native (no L2-normalization needed, unlike the angular
    # datasets below). 1,000,000 train vectors by default.

    # KMeans, full dataset, few clusters: ~3s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 10, "random_state": 0},
        "tier": "fast",
    }
    # KMeans, full dataset, many clusters: ~29s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 100, "random_state": 0},
    }
    # n_samples/n_clusters ~= 100 (10000/100).
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 1000, "random_state": 0},
        "split_kwargs": {"train_size": 100000, "test_size": 1000},
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
        "estimator_params": {"n_clusters": 10, "random_state": 0},
    }
    # KMeans, full dataset, many clusters: ~15s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 100, "random_state": 0},
        "tier": "normal",
    }
    # Small subsample via split_kwargs: a near-instant sanity-check case for
    # validating the config wiring, not a realistic workload.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 5, "random_state": 0},
        "split_kwargs": {"train_size": 2000, "test_size": 200},
        "tier": "test",
    }


@real_case_dataset("fashion_mnist_784", "clustering", "normal")
def fashion_mnist_784(implem: dict):
    # ANN-benchmarks vertical: Fashion-MNIST images flattened to 784-dim
    # pixel vectors, Euclidean distance native. 60,000 train vectors by
    # default. KMeans, full dataset, many clusters: ~7.5s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 100, "random_state": 0},
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
        "estimator_params": {"n_clusters": 10, "random_state": 0},
    }
    # KMeans, full dataset, many clusters: ~14s/fit.
    yield {
        "estimator": "KMeans",
        "estimator_params": {"n_clusters": 1000, "random_state": 0},
        "tier": "normal",
    }


def generate_cases(implem: dict | None = None, max_tier: str = "normal") -> list[dict]:
    max_tier_index = TIERS.index(max_tier)
    cases = []
    for case_func in REAL_DATASET_CASE_FUNCS:
        for case in case_func(implem):
            if TIERS.index(case["metadata"]["tier"]) <= max_tier_index:
                cases.append(case)
    return cases
