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

import os

from ...config import EstimatorCase
from .loaders import dataset_loading_functions, load_openml_data
from .loading import load_from_cache_or_compute
from .preprocessing import preprocess_data, preprocess_x
from .synthetic import generate_synthetic_data
from .transformer import convert_subsets, split_data


def load_data(bench_case: EstimatorCase) -> tuple[tuple, dict]:
    data_params = bench_case.data

    if data_params.generation_kwargs is not None:
        data_dict, data_description = generate_synthetic_data(
            function_name=data_params.source,
            generation_kwargs=data_params.generation_kwargs,
        )
        return convert_subsets(bench_case, data_dict, data_description)

    data_name = data_params.name(shortened=False)
    data_cache = os.environ.get("SKLBENCH_DATA_CACHE", "data_cache")
    raw_data_cache = os.path.join(data_cache, "raw")
    common_kwargs = {
        "data_name": data_name,
        "data_cache": data_cache,
        "raw_data_cache": raw_data_cache,
    }
    os.makedirs(data_cache, exist_ok=True)
    os.makedirs(raw_data_cache, exist_ok=True)

    if data_params.dataset is not None and data_params.dataset in dataset_loading_functions:
        data, data_description = load_from_cache_or_compute(
            dataset_loading_functions[data_params.dataset], **common_kwargs
        )
    elif data_params.source == "fetch_openml":
        data, data_description = load_from_cache_or_compute(
            load_openml_data, openml_id=data_params.id, **common_kwargs
        )
    else:
        raise ValueError(
            "Unable to get data from bench_case:\n"
            f"{data_params.model_dump(exclude_none=True)}"
        )

    preproc_kwargs = data_params.preprocessing_kwargs
    data = preprocess_data(data, **preproc_kwargs)
    data["x"] = preprocess_x(data["x"], **preproc_kwargs)

    data_dict, data_description = split_data(bench_case, data, data_description)
    return convert_subsets(bench_case, data_dict, data_description)
