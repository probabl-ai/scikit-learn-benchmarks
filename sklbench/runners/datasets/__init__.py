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

import pandas as pd

from ...config import EstimatorCase
from .loaders import dataset_loading_functions, load_openml_data
from .loading import load_from_cache_or_compute
from .preprocessing import build_transfer_to_device, split_and_preprocess_data
from .synthetic import generate_synthetic_data
from .transformer import convert_subsets


def load_raw_data(bench_case: EstimatorCase) -> tuple[dict, dict]:
    """Fetches or generates the case's raw dataset - the cacheable part of
    loading, done once per case (see `preprocess_data` for the rest).
    Returns `(raw_data, data_description)`.
    """
    data_params = bench_case.data

    if data_params.generation_kwargs is not None:
        return generate_synthetic_data(
            function_name=data_params.source,
            generation_kwargs=data_params.generation_kwargs,
        )

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
        return load_from_cache_or_compute(
            dataset_loading_functions[data_params.dataset], **common_kwargs
        )
    if data_params.source == "fetch_openml":
        return load_from_cache_or_compute(
            load_openml_data, openml_id=data_params.id, **common_kwargs
        )
    raise ValueError(
        "Unable to get data from bench_case:\n"
        f"{data_params.model_dump(exclude_none=True)}"
    )


def _shape_desc(data) -> dict:
    desc = {"samples": data.shape[0]}
    if len(data.shape) == 2:
        desc["features"] = data.shape[1]
    if isinstance(data, pd.DataFrame):
        n_categorical = int((data.dtypes == "category").sum())
        if n_categorical:
            desc["n_categorical_features"] = n_categorical
    return desc


def preprocess_data(
    bench_case: EstimatorCase, raw_data: dict, data_description: dict
) -> tuple[tuple, dict]:
    """Splits, encodes and transfers `raw_data` (from `load_raw_data`) to
    its target library/device/dtype/order. Called fresh on every repeat by
    `run_case_once` so the preprocessing pipeline itself is part of what's
    measured. Doesn't mutate `raw_data` or `data_description`.
    """
    data_params = bench_case.data

    if data_params.generation_kwargs is not None:
        data, description = convert_subsets(bench_case, dict(raw_data), dict(data_description))
        description["preprocessing"] = _shape_desc(raw_data["x_train"])
        return data, description

    implementation = bench_case.implementation
    preprocessing_defaults = data_description.get('preprocessing_defaults', {}).get(
        data_params.preprocessing_kind, {}
    )

    # A pipeline transformer only ever acts on X, so labels are transferred
    # separately below, with their own dtype (classification forces "int").
    transfer_to_device = build_transfer_to_device(
        dformat=implementation.data_library,
        device=implementation.device,
        dtype=data_params.dtype,
        order=data_params.order,
    )
    required_label_dtype = "int" if "n_classes" in data_description else None
    label_transfer_to_device = build_transfer_to_device(
        dformat=implementation.data_library,
        device=implementation.device,
        dtype=required_label_dtype or data_params.dtype,
        order=data_params.order,
    )

    preprocessing_kwargs = (
        preprocessing_defaults
        | data_params.preprocessing_kwargs
        | {"transfer_to_device": transfer_to_device}
    )

    data_dict = split_and_preprocess_data(
        raw_data,
        split_kwargs=data_params.split_kwargs,
        default_split=data_description.get('default_split'),
        preprocessing_kind=data_params.preprocessing_kind,
        preprocessing_kwargs=preprocessing_kwargs,
    )

    if data_dict["y_train"] is not None:
        data_dict["y_train"] = label_transfer_to_device.fit_transform(data_dict["y_train"])
        data_dict["y_test"] = label_transfer_to_device.transform(data_dict["y_test"])

    subset_description = dict(data_description)
    for subset_name in ("x_train", "x_test"):
        subset_description[subset_name] = {
            "format": implementation.data_library,
            "order": data_params.order,
            "dtype": data_params.dtype,
            **_shape_desc(data_dict[subset_name]),
        }
    subset_description["preprocessing"] = _shape_desc(raw_data["x"])

    return (
        tuple(data_dict[name] for name in ["x_train", "x_test", "y_train", "y_test"]),
        subset_description,
    )
