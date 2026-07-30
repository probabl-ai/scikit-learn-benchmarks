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

import hashlib
import json
import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NB: non-registered data components and extensions will not be found by loader
KNOWN_DATA_COMPONENTS = ["x", "y"]
KNOWN_DATA_EXTENSIONS = ["parq", "npz"]
PREPROCESSED_SUBSETS = ["x_train", "x_test", "y_train", "y_test"]


def get_expr_by_prefix(prefix: str) -> str:
    def get_or_expr_from_list(a: list[str]) -> str:
        # transforms list to OR expression: "['x', 'y']" -> "x|y"
        return str(a)[1:-1].replace("'", "").replace(", ", "|")

    data_comp_expr = get_or_expr_from_list(KNOWN_DATA_COMPONENTS)
    data_ext_expr = get_or_expr_from_list(KNOWN_DATA_EXTENSIONS)

    return f"{prefix}_({data_comp_expr}).({data_ext_expr})"


def get_filenames_by_prefix(directory: str, prefix: str) -> list[str]:
    assert os.path.isdir(directory)
    prefix_expr = get_expr_by_prefix(prefix)
    return list(
        filter(lambda x: re.search(prefix_expr, x) is not None, os.listdir(directory))
    )


def load_data_file(filepath, extension):
    if extension == "parq":
        data = pd.read_parquet(filepath)
    elif extension == "npz":
        npz_content = np.load(filepath)
        data = npz_content["arr_0"]
        npz_content.close()
    else:
        raise ValueError(f'Unknown extension "{extension}"')
    return data


def load_data_from_cache(
    data_cache: str, data_name: str, categorical_columns: dict[str, list[str]] | None = None
) -> dict:
    # data filename format:
    # {data_name}_{data_component}.{file_ext}
    categorical_columns = categorical_columns or {}
    data_filenames = get_filenames_by_prefix(data_cache, data_name)
    data = dict()
    for data_filename in data_filenames:
        if data_filename.endswith(".json"):
            continue
        postfix = data_filename.replace(data_name, "")[1:]
        component, file_ext = postfix.split(".", 1)
        component_data = load_data_file(
            os.path.join(data_cache, data_filename), file_ext
        )
        # parquet round-trips lose (fastparquet<->pyarrow) or corrupt
        # (fastparquet<->fastparquet) pandas `category` dtype, so it's
        # restored here from the columns recorded at cache-write time.
        for column in categorical_columns.get(component, []):
            if column in component_data.columns:
                component_data[column] = component_data[column].astype("category")
        data[component] = component_data
    return data


def save_data_to_cache(data: dict, data_cache: str, data_name: str) -> dict[str, list[str]]:
    categorical_columns = {}
    for component_name, data_compoment in data.items():
        component_filepath = os.path.join(data_cache, f"{data_name}_{component_name}")
        # convert 2d numpy array to pandas DataFrame for better caching
        if isinstance(data_compoment, np.ndarray) and data_compoment.ndim == 2:
            data_compoment = pd.DataFrame(data_compoment)
        # branching by data type for saving to cache
        if isinstance(data_compoment, pd.DataFrame):
            component_filepath += ".parq"
            data_compoment.columns = [
                column if isinstance(column, str) else str(column)
                for column in list(data_compoment.columns)
            ]
            component_categorical_columns = list(
                data_compoment.columns[data_compoment.dtypes == "category"]
            )
            if component_categorical_columns:
                categorical_columns[component_name] = component_categorical_columns
            data_compoment.to_parquet(
                component_filepath, engine="fastparquet", compression="snappy"
            )
        elif isinstance(data_compoment, pd.Series):
            component_filepath += ".npz"
            np.savez(component_filepath, data_compoment.to_numpy())
        elif isinstance(data_compoment, np.ndarray):
            component_filepath += ".npz"
            np.savez(component_filepath, data_compoment)
    return categorical_columns


def load_data_description(data_cache: str, data_name: str) -> dict:
    with open(os.path.join(data_cache, f"{data_name}.json"), "r") as desc_file:
        data_desc = json.load(desc_file)
    return data_desc


def save_data_description(data_desc: dict, data_cache: str, data_name: str):
    with open(os.path.join(data_cache, f"{data_name}.json"), "w") as desc_file:
        json.dump(data_desc, desc_file)


def load_from_cache_or_compute(
    function, *, data_name: str, data_cache: str, raw_data_cache: str, **extra_kwargs
) -> tuple[dict, dict]:
    if len(get_filenames_by_prefix(data_cache, data_name)) > 0:
        logger.info(f'Loading "{data_name}" dataset from cache files')
        data_desc = load_data_description(data_cache, data_name)
        data = load_data_from_cache(
            data_cache, data_name, categorical_columns=data_desc.get("categorical_columns")
        )
    else:
        logger.info(f'Loading "{data_name}" dataset from scratch')
        data, data_desc = function(raw_data_cache=raw_data_cache, **extra_kwargs)
        categorical_columns = save_data_to_cache(data, data_cache, data_name)
        if categorical_columns:
            data_desc["categorical_columns"] = categorical_columns
        save_data_description(data_desc, data_cache, data_name)
    return data, data_desc


def load_preprocessed_from_cache(
    data_cache: str,
    cache_name: str,
    categorical_columns: dict[str, list[str]] | None = None,
) -> dict:
    # unlike `load_data_from_cache`, subset names (x_train, x_test, ...) aren't
    # in `KNOWN_DATA_COMPONENTS`, so files are looked up directly by name
    # rather than discovered via `get_filenames_by_prefix`
    categorical_columns = categorical_columns or {}
    data_dict = {}
    for subset_name in PREPROCESSED_SUBSETS:
        subset_filepath = os.path.join(data_cache, f"{cache_name}_{subset_name}")
        if os.path.exists(subset_filepath + ".parq"):
            subset_data = load_data_file(subset_filepath + ".parq", "parq")
            for column in categorical_columns.get(subset_name, []):
                if column in subset_data.columns:
                    subset_data[column] = subset_data[column].astype("category")
        elif os.path.exists(subset_filepath + ".npz"):
            subset_data = load_data_file(subset_filepath + ".npz", "npz")
        else:
            subset_data = None
        data_dict[subset_name] = subset_data
    return data_dict


def get_preprocessing_cache_name(
    data_name: str,
    split_kwargs: dict,
    preprocessing_kind: str | None,
    preprocessing_kwargs: dict,
) -> str:
    key_repr = json.dumps(
        {
            "split_kwargs": split_kwargs,
            "preprocessing_kind": preprocessing_kind,
            "preprocessing_kwargs": preprocessing_kwargs,
        },
        sort_keys=True,
        default=str,
    )
    key_hash = hashlib.sha256(key_repr.encode("utf-8")).hexdigest()[:8]
    return f"{data_name}_{key_hash}"


def load_preprocessed_from_cache_or_compute(
    function,
    *,
    data_name: str,
    preprocessed_data_cache: str,
    split_kwargs: dict,
    preprocessing_kind: str | None,
    preprocessing_kwargs: dict,
) -> dict:
    """Cache the post-split, post-preprocessing x_train/x_test/y_train/y_test
    subsets, so repeated runs of the same case skip both loading and
    preprocessing. Keyed on the raw dataset name plus the split/preprocessing
    settings that produced the subsets, so cases sharing a dataset but using
    different splits or preprocessing don't collide.

    The cache key covers `split_kwargs`/`preprocessing_kind`/
    `preprocessing_kwargs` only, not the preprocessing code itself (same
    limitation as `load_from_cache_or_compute`'s raw-data cache). Changing
    what `trees_preprocessor`/`linear_preprocessor`/`hgb_preprocessing` (in
    `preprocessing.py`) actually do will not change the cache key, so stale
    subsets from before the change can still be served. Clear
    `{data_cache}/preprocessed/` manually after editing those functions.
    """
    cache_name = get_preprocessing_cache_name(
        data_name, split_kwargs, preprocessing_kind, preprocessing_kwargs
    )
    cache_desc_path = os.path.join(preprocessed_data_cache, f"{cache_name}.json")
    if os.path.exists(cache_desc_path):
        logger.info(f'Loading preprocessed "{data_name}" dataset from cache files')
        data_desc = load_data_description(preprocessed_data_cache, cache_name)
        return load_preprocessed_from_cache(
            preprocessed_data_cache,
            cache_name,
            categorical_columns=data_desc.get("categorical_columns"),
        )

    logger.info(f'Preprocessing "{data_name}" dataset from scratch')
    data_dict = function()
    categorical_columns = save_data_to_cache(data_dict, preprocessed_data_cache, cache_name)
    data_desc = {"categorical_columns": categorical_columns} if categorical_columns else {}
    save_data_description(data_desc, preprocessed_data_cache, cache_name)
    return data_dict
