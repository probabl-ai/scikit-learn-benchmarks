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
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.utils import check_random_state


ColumnSpec = str | Sequence[str]


def transform_columns(
    X: np.ndarray, columns: str, rng: np.random.RandomState,
    as_frame=False,
) -> None:
    n_features = X.shape[1]
    if columns == "mix":
        columns = ["continuous", "binary", "long-tail"] * n_features
        columns = columns[:n_features]
    elif isinstance(columns, str):
        columns = [columns] * n_features

    for i, col_idx in enumerate(rng.permutation(n_features)):
        col_type = columns[i]
        values = X[:, col_idx]

        if col_type == "continuous":
            continue
        if col_type == "binary":
            thresholds = np.quantile(values, np.sort(rng.uniform(size=2)))
            X[:, col_idx] = np.searchsorted(thresholds, values) == 1
        elif col_type == "long-tail":
            noise_ratio = rng.uniform(0.05, 0.5)
            n_bins = max(1, min(3, rng.poisson(10)))
            binned = np.searchsorted(
                np.linspace(values.min() + 1e-10, values.max(), num=n_bins),
                values,
            )
            mask = rng.rand(binned.size) < noise_ratio
            if mask.any():
                binned[mask] = rng.geometric(min(1, 20 / mask.sum()), size=mask.sum())
            X[:, col_idx] = binned
        else:
            raise ValueError(col_type)

    if not as_frame:
        return X

    X = pd.DataFrame(X)
    for i in range(n_features):
        n_uniques = len(X[i].value_counts())
        if 2 < n_uniques < 255:
            X[i] = X[i].astype("category")

    return X


def make_trees_regression_data(*, columns: ColumnSpec, random_state=None, as_frame=False, **kwargs):
    rng = check_random_state(random_state)
    X, y = make_regression(**kwargs, random_state=rng)
    X = transform_columns(X, columns, rng, as_frame=as_frame)
    return X, y


def make_trees_classification_data(*, columns: ColumnSpec, random_state=None, as_frame=False, **kwargs):
    rng = check_random_state(random_state)
    X, y = make_classification(**kwargs, random_state=rng)
    X = transform_columns(X, columns, rng, as_frame=as_frame)
    return X, y
