import numpy as np
import pandas as pd
import pytest

from sklbench.runners.datasets.synthetic import (
    make_trees_classification_data,
    make_trees_regression_data,
    tree_synthetic_transform,
)


def test_transform_columns_keeps_continuous_columns_unchanged():
    X = np.arange(12, dtype=float).reshape(4, 3)
    original = X.copy()

    transformed = tree_synthetic_transform(X, "continuous", np.random.RandomState(0))

    assert transformed is X
    np.testing.assert_array_equal(X, original)


def test_transform_columns_binarizes_columns():
    X = np.random.RandomState(42).normal(size=(40, 3))

    transformed = tree_synthetic_transform(X, "binary", np.random.RandomState(0))

    assert transformed is X
    assert set(np.unique(X)) <= {0.0, 1.0}
    assert all(set(np.unique(X[:, i])) == {0.0, 1.0} for i in range(X.shape[1]))


def test_transform_columns_generates_long_tail_integer_bins():
    X = np.random.RandomState(42).normal(size=(40, 3))

    tree_synthetic_transform(X, "long-tail", np.random.RandomState(0))

    assert np.all(X >= 0)
    assert np.all(X == X.astype(int))
    assert all(1 < len(np.unique(X[:, i])) < 40 for i in range(X.shape[1]))


def test_transform_columns_accepts_sequence_column_spec():
    X = np.tile(np.arange(12, dtype=float), (3, 1)).T
    original = X.copy()

    tree_synthetic_transform(
        X,
        ["continuous", "binary", "long-tail"],
        np.random.RandomState(0),
    )

    assert any(np.array_equal(X[:, i], original[:, i]) for i in range(X.shape[1]))
    assert any(set(np.unique(X[:, i])) <= {0.0, 1.0} for i in range(X.shape[1]))
    assert any(len(np.unique(X[:, i])) > 2 for i in range(X.shape[1]))


def test_transform_columns_returns_dataframe_with_categorical_long_tail_columns():
    X = np.random.RandomState(42).normal(size=(40, 3))

    transformed = tree_synthetic_transform(
        X,
        "long-tail",
        np.random.RandomState(2),
        as_frame=True,
    )

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape == (40, 3)
    assert all(
        isinstance(transformed[i].dtype, pd.CategoricalDtype)
        for i in transformed.columns
    )


def test_transform_columns_rejects_unknown_column_type():
    X = np.arange(12, dtype=float).reshape(4, 3)

    with pytest.raises(ValueError, match="unknown"):
        tree_synthetic_transform(X, "unknown", np.random.RandomState(0))


def test_make_trees_regression_data_uses_transform_columns():
    X, y = make_trees_regression_data(
        columns="binary",
        random_state=0,
        n_samples=30,
        n_features=4,
    )

    assert X.shape == (30, 4)
    assert y.shape == (30,)
    assert set(np.unique(X)) <= {0.0, 1.0}


def test_make_trees_classification_data_can_return_dataframe():
    X, y = make_trees_classification_data(
        columns="long-tail",
        random_state=0,
        as_frame=True,
        n_samples=30,
        n_features=5,
        n_informative=3,
        n_redundant=0,
    )

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (30, 5)
    assert y.shape == (30,)
    assert set(np.unique(y)) == {0, 1}
    assert any(isinstance(X[i].dtype, pd.CategoricalDtype) for i in X.columns)
