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
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)

from ...utils.custom_types import Array

logger = logging.getLogger(__name__)


def preprocess_data(
    data_dict: dict[str, Array],
    subsample: float | int | None = None,
    **kwargs,
) -> dict[str, Array]:
    """Preprocessing function applied for all data arguments."""
    if subsample is not None:
        for data_name, data in data_dict.items():
            data_dict[data_name] = train_test_split(
                data, train_size=subsample, random_state=42, shuffle=True
            )[0]
    return data_dict


def preprocess_x(
    x: Array,
    replace_nan="auto",
    category_encoding="ordinal",
    normalize=None,
    force_for_sparse=True,
    **kwargs,
) -> Array:
    """Preprocessing function applied only for `x` data argument."""
    return_type = type(x)
    if force_for_sparse and isinstance(x, csr_matrix):
        x = x.toarray()
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if not isinstance(x, pd.DataFrame):
        logger.warning(
            "Preprocessing is supported only for pandas DataFrames "
            f"and numpy ndarray. Got {type(x)} instead."
        )
        return x
    # NaN values replacement
    if x.isna().any().any():
        nan_columns = x.columns[x.isna().any(axis=0)]
        nan_df = x[nan_columns]
        if replace_nan == "auto":
            replace_nan = "median"
            logger.debug(f'Changing "replace_nan" from "auto" to "{replace_nan}".')
        if replace_nan == "median":
            nan_df = nan_df.fillna(nan_df.median())
        elif replace_nan == "mean":
            nan_df = nan_df.fillna(nan_df.mean())
        elif replace_nan == "ignore":
            pass
        else:
            logger.warning(f'Unknown "{replace_nan}" replace nan type.')
        x[nan_columns] = nan_df
    # Categorical features transformation
    categ_columns = x.columns[(x.dtypes == "category") + (x.dtypes == "object")]
    if len(categ_columns) > 0:
        if category_encoding == "onehot":
            prev_n_columns = x.shape[1]
            x = pd.get_dummies(x, columns=list(categ_columns))
            logger.debug(
                f"OneHotEncoder extended {prev_n_columns} columns to {x.shape[1]}."
            )
        elif category_encoding == "ordinal":
            encoder = OrdinalEncoder()
            encoder.set_output(transform="pandas")
            ordinal_df = encoder.fit_transform(x[categ_columns])
            x = x.drop(columns=categ_columns).join(ordinal_df)
        elif category_encoding == "drop":
            x = x.drop(columns=categ_columns)
        elif category_encoding == "ignore":
            pass
        else:
            logger.warning(f'Unknown "{category_encoding}" category encoding type.')
    # Normalization
    if normalize:
        if normalize == "standard":
            scaler = StandardScaler(with_mean=True, with_std=True)
        elif normalize == "mean":
            scaler = StandardScaler(with_mean=True, with_std=False)
        elif normalize == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            logger.warning(f'Unknown "{normalize}" normalization type.')
        if scaler is not None:
            x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
    if return_type == np.ndarray:
        return np.array(x)
    else:
        return x
