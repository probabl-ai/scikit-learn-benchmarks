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

import json
import logging
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# NB: non-registered data components and extensions will not be found by loader
KNOWN_DATA_COMPONENTS = ["x", "y"]
KNOWN_DATA_EXTENSIONS = ["parq", "npz", "csr.npz"]


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
    elif extension.endswith("npz"):
        npz_content = np.load(filepath)
        if extension == "npz":
            data = npz_content["arr_0"]
        elif extension == "csr.npz":
            data = csr_matrix(
                tuple(npz_content[attr] for attr in ["data", "indices", "indptr"])
            )
        else:
            raise ValueError(f'Unknown npz subextension "{extension}"')
        npz_content.close()
    else:
        raise ValueError(f'Unknown extension "{extension}"')
    return data


def load_data_from_cache(data_cache: str, data_name: str) -> dict:
    # data filename format:
    # {data_name}_{data_component}.{file_ext}
    data_filenames = get_filenames_by_prefix(data_cache, data_name)
    data = dict()
    for data_filename in data_filenames:
        if data_filename.endswith(".json"):
            continue
        postfix = data_filename.replace(data_name, "")[1:]
        component, file_ext = postfix.split(".", 1)
        data[component] = load_data_file(
            os.path.join(data_cache, data_filename), file_ext
        )
    return data


def save_data_to_cache(data: dict, data_cache: str, data_name: str):
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
            data_compoment.to_parquet(
                component_filepath, engine="fastparquet", compression="snappy"
            )
        elif isinstance(data_compoment, csr_matrix):
            component_filepath += ".csr.npz"
            np.savez(
                component_filepath,
                **{
                    attr: getattr(data_compoment, attr)
                    for attr in ["data", "indices", "indptr"]
                },
            )
        elif isinstance(data_compoment, pd.Series):
            component_filepath += ".npz"
            np.savez(component_filepath, data_compoment.to_numpy())
        elif isinstance(data_compoment, np.ndarray):
            component_filepath += ".npz"
            np.savez(component_filepath, data_compoment)


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
        data = load_data_from_cache(data_cache, data_name)
        data_desc = load_data_description(data_cache, data_name)
    else:
        logger.info(f'Loading "{data_name}" dataset from scratch')
        data, data_desc = function(raw_data_cache=raw_data_cache, **extra_kwargs)
        save_data_to_cache(data, data_cache, data_name)
        save_data_description(data_desc, data_cache, data_name)
    return data, data_desc
