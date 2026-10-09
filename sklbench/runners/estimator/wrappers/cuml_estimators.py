"""Native cuML estimators behind sklearn-like constructor APIs.

Configs and the estimator runner pass scikit-learn hyper-parameter names
(e.g. ``solver="lbfgs"``, ``criterion="gini"``, ``max_leaf_nodes``). cuML uses
overlapping but not identical names and defaults; these wrappers remap the
common cases and force ``output_type="numpy"`` so the harness gets host arrays.

Import of cuML is deferred until a wrapper class is first resolved, so other
Pixi environments can import ``loading`` without having RAPIDS installed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LogisticRegression",
    "Ridge",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "KMeans",
    "map_logistic_regression_params",
    "map_ridge_params",
    "map_random_forest_params",
    "map_kmeans_params",
]


# Sklearn solver names that configs pass for LogisticRegression; cuML only
# supports the Quasi-Newton solver ``qn`` (L-BFGS or OWL-QN under the hood).
_LR_SOLVER_TO_CUML = {
    "qn": "qn",
    "lbfgs": "qn",
    "newton-cg": "qn",
    "newton-cholesky": "qn",
    "sag": "qn",
    "saga": "qn",
    "liblinear": "qn",
}

# From cuml.linear_model.ridge._SOLVER_SKLEARN_TO_CUML
_RIDGE_SOLVER_TO_CUML = {
    "auto": "auto",
    "svd": "svd",
    "eig": "eig",
    "cholesky": "eig",
    "lsqr": "eig",
    "sag": "eig",
    "saga": "eig",
    "sparse_cg": "eig",
}

_RF_CRITERION_TO_SPLIT = {
    "gini": "gini",
    "entropy": "entropy",
    "log_loss": "entropy",
    "squared_error": "mse",
    "poisson": "poisson",
    # Already-cuML names:
    "mse": "mse",
    "mae": "mae",
}

# Sklearn-only / CPU-threading knobs the harness may pass; drop rather than
# fail when constructing a GPU estimator.
_RF_DROP_KEYS = frozenset(
    {
        "n_jobs",
        "warm_start",
        "oob_score",
        "class_weight",
        "ccp_alpha",
        "min_weight_fraction_leaf",
        "monotonic_cst",
        "verbose",
    }
)

_KMEANS_DROP_KEYS = frozenset(
    {
        "algorithm",  # lloyd/elkan — CPU-only sklearn detail
        "copy_x",
        "n_jobs",
        "verbose",
    }
)


def map_logistic_regression_params(params: dict[str, Any]) -> dict[str, Any]:
    """Map sklearn-like LogisticRegression kwargs to cuML's constructor."""
    out = dict(params)
    solver = out.pop("solver", "lbfgs")
    if solver not in _LR_SOLVER_TO_CUML:
        raise ValueError(
            f"Unsupported LogisticRegression solver for cuML: {solver!r}. "
            f"Expected one of {sorted(_LR_SOLVER_TO_CUML)}."
        )
    out["solver"] = _LR_SOLVER_TO_CUML[solver]

    # cuML has no warm_start / intercept_scaling / multi_class / n_jobs / random_state
    for key in (
        "warm_start",
        "intercept_scaling",
        "multi_class",
        "n_jobs",
        "random_state",
        "dual",
        "verbose",
    ):
        out.pop(key, None)

    out.setdefault("output_type", "numpy")
    return out


def map_ridge_params(params: dict[str, Any]) -> dict[str, Any]:
    """Map sklearn-like Ridge kwargs to cuML's constructor."""
    out = dict(params)
    solver = out.pop("solver", "auto")
    if solver not in _RIDGE_SOLVER_TO_CUML:
        raise ValueError(
            f"Unsupported Ridge solver for cuML: {solver!r}. "
            f"Expected one of {sorted(_RIDGE_SOLVER_TO_CUML)}."
        )
    out["solver"] = _RIDGE_SOLVER_TO_CUML[solver]

    if out.pop("positive", False):
        raise ValueError("cuML Ridge does not support positive=True")

    for key in ("tol", "max_iter", "random_state", "copy_X", "verbose"):
        out.pop(key, None)

    out.setdefault("output_type", "numpy")
    return out


def map_random_forest_params(
    params: dict[str, Any], *, default_split_criterion: str
) -> dict[str, Any]:
    """Map sklearn-like RandomForest kwargs to cuML's constructor."""
    out = dict(params)
    for key in _RF_DROP_KEYS:
        out.pop(key, None)

    if "criterion" in out:
        criterion = out.pop("criterion")
        if criterion not in _RF_CRITERION_TO_SPLIT:
            raise ValueError(
                f"Unsupported RandomForest criterion for cuML: {criterion!r}."
            )
        out["split_criterion"] = _RF_CRITERION_TO_SPLIT[criterion]
    else:
        out.setdefault("split_criterion", default_split_criterion)

    if "max_leaf_nodes" in out:
        max_leaf_nodes = out.pop("max_leaf_nodes")
        out["max_leaves"] = -1 if max_leaf_nodes is None else max_leaf_nodes

    # sklearn max_depth=None means unlimited; cuML's default is 16 when omitted.
    # Only forward an explicit finite depth (same rule as cuML's _params_from_cpu).
    if out.get("max_depth") is None:
        out.pop("max_depth", None)

    if isinstance(out.get("max_samples"), int):
        raise ValueError(
            "cuML RandomForest does not support integer max_samples; "
            "pass a float in (0, 1]."
        )

    out.setdefault("output_type", "numpy")
    return out


def map_kmeans_params(params: dict[str, Any]) -> dict[str, Any]:
    """Map sklearn-like KMeans kwargs to cuML's constructor."""
    out = dict(params)
    for key in _KMEANS_DROP_KEYS:
        out.pop(key, None)

    init = out.get("init", "k-means++")
    if isinstance(init, str):
        if init == "k-means++":
            out["init"] = "scalable-k-means++"
        elif init in ("scalable-k-means++", "k-means||", "random"):
            out["init"] = "k-means||" if init == "k-means||" else init
        else:
            raise ValueError(f"Unsupported KMeans init for cuML: {init!r}.")

    out.setdefault("output_type", "numpy")
    return out


def _build_logistic_regression():
    from cuml.linear_model import LogisticRegression as _CumlLogisticRegression

    class LogisticRegression(_CumlLogisticRegression):
        def __init__(
            self,
            *,
            penalty="l2",
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            class_weight=None,
            max_iter=1000,
            linesearch_max_iter=50,
            l1_ratio=None,
            solver="lbfgs",
            warm_start=False,
            intercept_scaling=1,
            multi_class="deprecated",
            n_jobs=None,
            random_state=None,
            verbose=False,
            handle=None,
            output_type="numpy",
        ):
            mapped = map_logistic_regression_params(
                {
                    "penalty": penalty,
                    "tol": tol,
                    "C": C,
                    "fit_intercept": fit_intercept,
                    "class_weight": class_weight,
                    "max_iter": max_iter,
                    "linesearch_max_iter": linesearch_max_iter,
                    "l1_ratio": l1_ratio,
                    "solver": solver,
                    "warm_start": warm_start,
                    "intercept_scaling": intercept_scaling,
                    "multi_class": multi_class,
                    "n_jobs": n_jobs,
                    "random_state": random_state,
                    "verbose": verbose,
                    "handle": handle,
                    "output_type": output_type,
                }
            )
            super().__init__(**mapped)

    return LogisticRegression


def _build_ridge():
    from cuml.linear_model import Ridge as _CumlRidge

    class Ridge(_CumlRidge):
        def __init__(
            self,
            *,
            alpha=1.0,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            max_iter=None,
            tol=1e-4,
            solver="auto",
            positive=False,
            random_state=None,
            handle=None,
            verbose=False,
            output_type="numpy",
        ):
            mapped = map_ridge_params(
                {
                    "alpha": alpha,
                    "fit_intercept": fit_intercept,
                    "normalize": normalize,
                    "copy_X": copy_X,
                    "max_iter": max_iter,
                    "tol": tol,
                    "solver": solver,
                    "positive": positive,
                    "random_state": random_state,
                    "handle": handle,
                    "verbose": verbose,
                    "output_type": output_type,
                }
            )
            super().__init__(**mapped)

    return Ridge


def _build_random_forest_classifier():
    from cuml.ensemble import RandomForestClassifier as _CumlRFC

    class RandomForestClassifier(_CumlRFC):
        def __init__(
            self,
            *,
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=False,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            n_bins=128,
            handle=None,
            output_type="numpy",
            **kwargs,
        ):
            params = {
                "n_estimators": n_estimators,
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "min_weight_fraction_leaf": min_weight_fraction_leaf,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "min_impurity_decrease": min_impurity_decrease,
                "bootstrap": bootstrap,
                "oob_score": oob_score,
                "n_jobs": n_jobs,
                "random_state": random_state,
                "verbose": verbose,
                "warm_start": warm_start,
                "class_weight": class_weight,
                "ccp_alpha": ccp_alpha,
                "n_bins": n_bins,
                "handle": handle,
                "output_type": output_type,
                **kwargs,
            }
            if max_samples is not None:
                params["max_samples"] = max_samples
            mapped = map_random_forest_params(
                params, default_split_criterion="gini"
            )
            super().__init__(**mapped)

    return RandomForestClassifier


def _build_random_forest_regressor():
    from cuml.ensemble import RandomForestRegressor as _CumlRFR

    class RandomForestRegressor(_CumlRFR):
        def __init__(
            self,
            *,
            n_estimators=100,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=False,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            n_bins=128,
            handle=None,
            output_type="numpy",
            **kwargs,
        ):
            params = {
                "n_estimators": n_estimators,
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "min_weight_fraction_leaf": min_weight_fraction_leaf,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "min_impurity_decrease": min_impurity_decrease,
                "bootstrap": bootstrap,
                "oob_score": oob_score,
                "n_jobs": n_jobs,
                "random_state": random_state,
                "verbose": verbose,
                "warm_start": warm_start,
                "ccp_alpha": ccp_alpha,
                "n_bins": n_bins,
                "handle": handle,
                "output_type": output_type,
                **kwargs,
            }
            if max_samples is not None:
                params["max_samples"] = max_samples
            mapped = map_random_forest_params(
                params, default_split_criterion="mse"
            )
            super().__init__(**mapped)

    return RandomForestRegressor


def _build_kmeans():
    from cuml.cluster import KMeans as _CumlKMeans

    class KMeans(_CumlKMeans):
        def __init__(
            self,
            *,
            n_clusters=8,
            init="k-means++",
            n_init="auto",
            max_iter=300,
            tol=1e-4,
            verbose=False,
            random_state=None,
            copy_x=True,
            algorithm="lloyd",
            handle=None,
            output_type="numpy",
            oversampling_factor=2.0,
            max_samples_per_batch=1 << 15,
        ):
            mapped = map_kmeans_params(
                {
                    "n_clusters": n_clusters,
                    "init": init,
                    "n_init": n_init,
                    "max_iter": max_iter,
                    "tol": tol,
                    "verbose": verbose,
                    "random_state": random_state,
                    "copy_x": copy_x,
                    "algorithm": algorithm,
                    "handle": handle,
                    "output_type": output_type,
                    "oversampling_factor": oversampling_factor,
                    "max_samples_per_batch": max_samples_per_batch,
                }
            )
            super().__init__(**mapped)

    return KMeans


_BUILDERS = {
    "LogisticRegression": _build_logistic_regression,
    "Ridge": _build_ridge,
    "RandomForestClassifier": _build_random_forest_classifier,
    "RandomForestRegressor": _build_random_forest_regressor,
    "KMeans": _build_kmeans,
}

_CACHE: dict[str, type] = {}


def __getattr__(name: str):
    if name in _BUILDERS:
        if name not in _CACHE:
            _CACHE[name] = _BUILDERS[name]()
        return _CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
