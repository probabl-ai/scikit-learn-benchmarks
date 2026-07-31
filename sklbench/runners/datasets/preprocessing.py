# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    TargetEncoder,
)
from sklearn.kernel_approximation import Nystroem


logger = logging.getLogger(__name__)

Array = pd.DataFrame | np.ndarray


def split_and_preprocess_data(
    data_dict: dict[str, Array],
    split_kwargs: dict | None = None,
    default_split: dict | None = None,
    preprocessing_kind: str | None = None,
    preprocessing_kwargs: dict | None = None,
) -> dict[str, Array]:
    """Preprocessing function applied for all data arguments."""
    data_dict = split_data(data_dict, split_kwargs, default_split)
    if preprocessing_kind is None:
        return data_dict
    preprocessing_func = PREPROCESSINGS[preprocessing_kind]
    data_dict['x_train'], data_dict['x_test'] = preprocessing_func(
        data_dict['x_train'], data_dict['x_test'], data_dict['y_train'],
        **preprocessing_kwargs
    )
    return data_dict


def train_test_split_wrapper(*args, **kwargs):
    if "ignore" in kwargs:
        result = []
        for arg in args:
            result += [arg, arg]
        return result
    else:
        return train_test_split(*args, **kwargs)


def split_data(
    data: dict, split_kwargs: dict | None, default_split: dict | None
) -> tuple[dict, dict]:
    """Split loaded `{"x": ..., "y": ...}` data into train/test subsets.

    Uses the dataset's own `default_split` (set by individual loaders) as a
    base, overridden by the case's `split_kwargs`.

    `default_split` is JSON-serialized to disk, so it cannot carry the actual
    `y` array for stratification. A loader that needs a stratified split sets
    `"stratify": "y"` as a string sentinel instead; it is resolved here to
    the real `y` array before being passed to `train_test_split`.
    """
    kwargs = (default_split or {}) | (split_kwargs or {})
    kwargs.setdefault("random_state", 42)

    x = data["x"]
    if "y" in data:
        y = data["y"]
        if kwargs.get("stratify") == "y":
            kwargs["stratify"] = y
        x_train, x_test, y_train, y_test = train_test_split_wrapper(x, y, **kwargs)
    else:
        x_train, x_test = train_test_split_wrapper(x, **kwargs)
        y_train, y_test = None, None

    data_dict = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return data_dict


def preprocessor_to_preprocessing(f):

    def preprocesing(X_train, X_test, y_train=None, **kwargs):
        preprocessor = f(**kwargs)
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test

    return preprocesing


@preprocessor_to_preprocessing
def trees_preprocessor(encoding : str = "ordinal"):

    encoders = {
        "ordinal": OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=-1,
            min_frequency=5
        ),
        "one-hot": OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=10,
            min_frequency=5,
        ),
        # TODO? target encoding
    } 

    encoder = encoders[encoding]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "encoder",
                encoder,
                make_column_selector(dtype_include=["category", object]),
            ),
        ],
        remainder='passthrough'
    )

    # TODO? returning categorical type as done for HGB
    # useful for sklearn nightly (categorical support)

    return preprocessor


@preprocessor_to_preprocessing
def linear_preprocessor(
    nystroem = None,
    passthrough_columns = (),
    spline_kwargs = None,
):
    """
    `passthrough_columns` and `spline_kwargs` let a loader flag (via
    `data_desc["preprocessing_defaults"]["linear"]`, merged into these
    kwargs by `load_data`) that some numeric columns are already
    well-conditioned features that shouldn't be spline-expanded (e.g.
    `fraud`'s PCA components), and/or that the spline transform applied to
    the remaining numeric columns should use non-default knot settings.
    """

    target_encoder = TargetEncoder(
        target_type="auto",
        cv=5,
        shuffle=True,
        random_state=0,
    )

    one_hot_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        max_categories=10,
        min_frequency=5,
    )

    numeric_selector = make_column_selector(dtype_include=["number"])

    def numeric_non_passthrough_columns(x):
        return [c for c in numeric_selector(x) if c not in passthrough_columns]

    spline_kwargs = {"n_knots": 10, "degree": 2, "handle_missing": "zeros"} | (
        spline_kwargs or {}
    )

    transformers = [
        (
            "categorical",
            FeatureUnion([
                ("onehot", one_hot_encoder),
                ("target", target_encoder),
            ]),
            make_column_selector(dtype_include=["category", object]),
        ),
        (
            "numeric",
            SplineTransformer(**spline_kwargs),
            numeric_non_passthrough_columns,
        ),
    ]
    if passthrough_columns:
        transformers.append(("passthrough", "passthrough", list(passthrough_columns)))

    preprocessor = ColumnTransformer(transformers=transformers)

    if nystroem is None:
        # it's already scaled I think?
        # So it should be fine for fitting linear models
        return preprocessor

    nystroem = dict(kernel="poly", degree=2, n_components=300, random_state=721) | nystroem

    return make_pipeline(preprocessor, Nystroem(**nystroem))


def hgb_preprocessing(X_train, X_test, y_train=None):

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=-1,
        min_frequency=5,
        max_categories=252
    )

    categorical_columns = X_train.select_dtypes(["category", object]).columns.to_list()
    preprocessor = ColumnTransformer(
        transformers=[("encoder", encoder, categorical_columns)],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    for col in categorical_columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    return X_train, X_test


PREPROCESSINGS = {
    'trees': trees_preprocessor,
    'linear': linear_preprocessor,
    'hgb': hgb_preprocessing,
}
