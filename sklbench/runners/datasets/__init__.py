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

import gc
import os

from ...config import EstimatorCase
from ...utils.common import custom_format
from .loaders import (
    dataset_loading_functions,
    load_custom_data,
    load_openml_data,
    load_sklearn_synthetic_data,
)


def load_data(bench_case: EstimatorCase) -> tuple[dict, dict]:
    data_params = bench_case.data
    data_name = data_params.name(shortened=False)
    data_cache = data_params.cache_directory or os.environ.get(
        "SKLBENCH_DATA_CACHE", "data_cache"
    )
    raw_data_cache = data_params.raw_cache_directory or os.path.join(data_cache, "raw")
    common_kwargs = {
        "data_name": data_name,
        "data_cache": data_cache,
        "raw_data_cache": raw_data_cache,
    }
    preproc_kwargs = data_params.preprocessing_kwargs
    os.makedirs(data_cache, exist_ok=True)
    os.makedirs(raw_data_cache, exist_ok=True)

    if data_params.dataset is not None:
        if data_params.dataset in dataset_loading_functions:
            return dataset_loading_functions[data_params.dataset](
                **common_kwargs,
                preproc_kwargs=preproc_kwargs,
                dataset_params=data_params.dataset_kwargs,
            )
        return load_custom_data(**common_kwargs, preproc_kwargs=preproc_kwargs)

    if data_params.source is not None:
        if data_params.source.startswith("make_"):
            return load_sklearn_synthetic_data(
                function_name=data_params.source,
                input_kwargs=data_params.generation_kwargs,
                preproc_kwargs=preproc_kwargs,
                **common_kwargs,
            )
        if data_params.source == "fetch_openml":
            return load_openml_data(
                openml_id=data_params.id,
                preproc_kwargs=preproc_kwargs,
                **common_kwargs,
            )

    raise ValueError(
        "Unable to get data from bench_case:\n"
        f"{custom_format(data_params.model_dump(mode='json', exclude_none=True))}"
    )


def load_data_with_cleanup(bench_case: EstimatorCase):
    result = load_data(bench_case)
    del result
    gc.collect()
