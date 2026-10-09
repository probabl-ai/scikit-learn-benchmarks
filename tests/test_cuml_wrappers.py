"""Param mapping for native cuML wrappers (no GPU / RAPIDS required)."""

import pytest

from sklbench.runners.estimator.wrappers.cuml_estimators import (
    map_kmeans_params,
    map_logistic_regression_params,
    map_random_forest_params,
    map_ridge_params,
)


def test_logistic_regression_maps_sklearn_solvers_to_qn():
    mapped = map_logistic_regression_params({"solver": "lbfgs", "C": 0.5})
    assert mapped["solver"] == "qn"
    assert mapped["C"] == 0.5
    assert mapped["output_type"] == "numpy"


def test_logistic_regression_rejects_unknown_solver():
    with pytest.raises(ValueError, match="Unsupported LogisticRegression solver"):
        map_logistic_regression_params({"solver": "not-a-solver"})


def test_ridge_maps_cholesky_to_eig():
    mapped = map_ridge_params({"solver": "cholesky", "alpha": 2.0})
    assert mapped["solver"] == "eig"
    assert mapped["alpha"] == 2.0


def test_random_forest_maps_criterion_and_drops_n_jobs():
    mapped = map_random_forest_params(
        {
            "criterion": "gini",
            "n_jobs": -1,
            "max_leaf_nodes": 32,
            "max_depth": None,
            "n_estimators": 10,
        },
        default_split_criterion="gini",
    )
    assert mapped["split_criterion"] == "gini"
    assert mapped["max_leaves"] == 32
    assert "max_depth" not in mapped
    assert "n_jobs" not in mapped
    assert "criterion" not in mapped


def test_random_forest_regressor_squared_error():
    mapped = map_random_forest_params(
        {"criterion": "squared_error"},
        default_split_criterion="mse",
    )
    assert mapped["split_criterion"] == "mse"


def test_kmeans_maps_kmeanspp_init():
    mapped = map_kmeans_params(
        {"init": "k-means++", "n_clusters": 5, "algorithm": "lloyd", "n_init": 1}
    )
    assert mapped["init"] == "scalable-k-means++"
    assert mapped["n_clusters"] == 5
    assert mapped["n_init"] == 1
    assert "algorithm" not in mapped


def test_wrapped_estimators_register_cuml():
    from sklbench.runners.estimator.loading import wrapped_estimators

    for name in (
        "LogisticRegression",
        "Ridge",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "KMeans",
    ):
        assert ("cuml", name) in wrapped_estimators
