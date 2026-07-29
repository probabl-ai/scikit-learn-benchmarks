import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_openml

from sklbench.runners.datasets.preprocessing import (
    hgb_preprocessing,
    linear_preprocessor,
    split_and_preprocess_data,
    split_data,
    train_test_split_wrapper,
    trees_preprocessor,
)

# Ames Housing (openml id 42165): real dataset with both categorical columns
# (many as plain "object" dtype, as produced by this codebase's openml loader)
# and actual missing values in both categorical and numeric columns, so it
# exercises every code path in the preprocessing functions below.
OPENML_HOUSING_ID = 42165


@pytest.fixture(scope="module")
def housing_xy():
    x, y = fetch_openml(data_id=OPENML_HOUSING_ID, as_frame=True, return_X_y=True)
    return x, y.to_numpy(dtype=float)


@pytest.fixture
def housing_data(housing_xy):
    x, y = housing_xy
    return {"x": x, "y": y}


def test_split_data_uses_default_random_state():
    data = {"x": np.arange(20).reshape(10, 2), "y": np.arange(10)}

    split = split_data(data, split_kwargs=None, default_split=None)
    split_again = split_data(data, split_kwargs=None, default_split=None)

    np.testing.assert_array_equal(split["x_train"], split_again["x_train"])


def test_split_data_split_kwargs_override_default_split():
    data = {"x": np.arange(20).reshape(10, 2), "y": np.arange(10)}

    split = split_data(
        data,
        split_kwargs={"test_size": 0.3},
        default_split={"test_size": 0.5},
    )

    assert split["x_train"].shape[0] == 7
    assert split["x_test"].shape[0] == 3


def test_split_data_without_y_leaves_y_train_test_none():
    data = {"x": np.arange(20).reshape(10, 2)}

    split = split_data(data, split_kwargs=None, default_split=None)

    assert split["y_train"] is None
    assert split["y_test"] is None


def test_train_test_split_wrapper_ignore_duplicates_instead_of_splitting():
    x = np.arange(10)
    y = np.arange(10) * 2

    result = train_test_split_wrapper(x, y, ignore=True)

    assert len(result) == 4
    np.testing.assert_array_equal(result[0], x)
    np.testing.assert_array_equal(result[1], x)
    np.testing.assert_array_equal(result[2], y)
    np.testing.assert_array_equal(result[3], y)


def test_split_and_preprocess_data_without_preprocessing_kind_is_a_noop_split(
    housing_data,
):
    out = split_and_preprocess_data(
        dict(housing_data),
        split_kwargs={"test_size": 0.25},
        default_split=None,
        preprocessing_kind=None,
        preprocessing_kwargs={},
    )

    assert isinstance(out["x_train"], pd.DataFrame)
    assert out["x_train"]["MSZoning"].dtype == object


@pytest.mark.parametrize("encoding", ["ordinal", "one-hot"])
def test_trees_preprocessor_encodes_real_categorical_columns(housing_data, encoding):
    out = split_and_preprocess_data(
        dict(housing_data),
        split_kwargs={"test_size": 0.25, "random_state": 0},
        default_split=None,
        preprocessing_kind="trees",
        preprocessing_kwargs={"encoding": encoding},
    )

    x_train, x_test = out["x_train"], out["x_test"]
    assert np.issubdtype(np.asarray(x_train).dtype, np.number)
    assert np.issubdtype(np.asarray(x_test).dtype, np.number)
    assert x_train.shape[0] + x_test.shape[0] == len(housing_data["x"])
    assert x_train.shape[1] == x_test.shape[1]


def test_trees_preprocessor_one_hot_expands_more_columns_than_ordinal(housing_data):
    ordinal_x_train, _ = trees_preprocessor(
        housing_data["x"], housing_data["x"], housing_data["y"], encoding="ordinal"
    )
    one_hot_x_train, _ = trees_preprocessor(
        housing_data["x"], housing_data["x"], housing_data["y"], encoding="one-hot"
    )

    assert one_hot_x_train.shape[1] > ordinal_x_train.shape[1]


def test_hgb_preprocessing_preserves_original_column_names(housing_data):
    x_train, x_test = hgb_preprocessing(
        housing_data["x"], housing_data["x"], housing_data["y"]
    )

    assert set(x_train.columns) == set(housing_data["x"].columns)
    assert set(x_test.columns) == set(housing_data["x"].columns)


def test_hgb_preprocessing_encodes_categoricals_as_category_dtype(housing_data):
    x_train, x_test = hgb_preprocessing(
        housing_data["x"], housing_data["x"], housing_data["y"]
    )

    categorical_columns = housing_data["x"].select_dtypes(
        include=["category", object]
    ).columns
    for col in categorical_columns:
        assert isinstance(x_train[col].dtype, pd.CategoricalDtype)
        assert isinstance(x_test[col].dtype, pd.CategoricalDtype)


def test_linear_preprocessor_uses_target_encoder_and_fills_missing_values(
    housing_data,
):
    x_train, x_test = linear_preprocessor(
        housing_data["x"], housing_data["x"], housing_data["y"], nystroem="no"
    )

    assert np.issubdtype(np.asarray(x_train).dtype, np.number)
    assert not np.isnan(np.asarray(x_train, dtype=float)).any()
    assert not np.isnan(np.asarray(x_test, dtype=float)).any()
    assert x_train.shape[1] == x_test.shape[1]


def test_linear_preprocessor_requires_y_train_for_target_encoding(housing_data):
    with pytest.raises(ValueError):
        linear_preprocessor(
            housing_data["x"], housing_data["x"], None, nystroem="no"
        )


def test_split_and_preprocess_data_with_linear_kind_end_to_end(housing_data):
    out = split_and_preprocess_data(
        dict(housing_data),
        split_kwargs={"test_size": 0.25, "random_state": 0},
        default_split=None,
        preprocessing_kind="linear",
        preprocessing_kwargs={"nystroem": "no"},
    )

    x_train, x_test = out["x_train"], out["x_test"]
    assert x_train.shape[0] + x_test.shape[0] == len(housing_data["x"])
    assert x_train.shape[1] == x_test.shape[1]
    assert not np.isnan(np.asarray(x_train, dtype=float)).any()
