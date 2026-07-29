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
import warnings

import numpy as np
import pandas as pd

from ...config import EstimatorCase

logger = logging.getLogger(__name__)


def _torch_dtype(dtype: str | None):
    if dtype is None:
        return None

    import torch

    dtype_aliases = {
        "float": "float32",
        "double": "float64",
        "int": "int64",
        "long": "int64",
    }
    dtype_name = dtype_aliases.get(dtype, dtype)
    torch_dtype = getattr(torch, dtype_name, None)
    if torch_dtype is None:
        raise ValueError(f"Unknown torch dtype {dtype}")
    return torch_dtype


def convert_data(
    data,
    dformat: str | None = None,
    order: str | None = None,
    dtype: str | None = None,
    device: str = None,
):
    """Convert `data` to the requested library/order/dtype.

    A no-op unless `dformat`, `order` or `dtype` is given. `order` only applies to
    array libraries (numpy, dpnp, torch, ...) — it is ignored for pandas (and,
    once introduced, polars) since those formats have no equivalent concept.
    """
    if dformat is None and order is None and dtype is None:
        return data
    if dtype == "preserve":
        dtype = None

    if isinstance(data, (pd.DataFrame, pd.Series)):
        if dtype is not None:
            data = data.astype(dtype)
    else:
        if order == "F":
            data = np.asfortranarray(data, dtype=dtype)
        elif order == "C" or (order is None and dtype is not None):
            data = np.ascontiguousarray(data, dtype=dtype)
        elif order is not None:
            raise ValueError(f"Unknown data order {order}")

    if dformat is None:
        return data
    if dformat == "numpy":
        return data
    elif dformat == "pandas":
        if data.ndim == 1:
            return pd.Series(data)
        return pd.DataFrame(data)
    elif dformat == "dpnp":
        import dpnp

        return dpnp.array(data, dtype=dtype, order=order, device=device)
    elif dformat == "torch":
        import torch

        kwargs = {"device": device} if device is not None else {}
        torch_dtype = _torch_dtype(dtype)
        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype
        return torch.asarray(data, **kwargs)
    elif dformat == "dpctl":
        warnings.warn(
            "dpctl tensors are deprecated and support for them "
            "in scikit-learn_bench will be removed. "
            "Consider using dpnp arrays instead.",
            FutureWarning,
        )
        import dpctl.tensor

        return dpctl.tensor.asarray(data, dtype=dtype, order=order, device=device)
    elif dformat == "cudf":
        import cudf

        if data.ndim == 1:
            return cudf.Series(data)
        if order == "C":
            logger.warning("cudf.DataFrame is not compatible with C data order")
        return cudf.DataFrame(data)
    elif dformat == "cupy":
        import cupy

        return cupy.array(data)
    else:
        raise ValueError(f"Unknown data format {dformat}")


def convert_subsets(
    bench_case: EstimatorCase, data_dict: dict, data_description: dict
) -> tuple[tuple, dict]:
    """Common final step for both the synthetic and loaded paths.

    Converts each `x_train`/`x_test`/`y_train`/`y_test` subset to the requested
    library/device/order/dtype. A no-op for any subset that has nothing
    requested (no `implementation.data_library`, no `data.order`/`data.dtype`,
    no per-subset override).
    """
    data_params = bench_case.data
    device = bench_case.implementation.device
    common_data_format = bench_case.implementation.data_library
    common_data_order = data_params.order
    common_data_dtype = data_params.dtype

    if "n_classes" in data_description:
        required_label_dtype = "int"
    else:
        required_label_dtype = None

    for subset_name, subset_content in data_dict.items():
        if subset_content is None:
            continue
        is_label = subset_name.startswith("y")

        subset_options = getattr(data_params, subset_name) or {}
        data_format = subset_options.get("format", common_data_format)
        data_order = subset_options.get("order", common_data_order)
        data_dtype = subset_options.get("dtype", common_data_dtype)

        if is_label and required_label_dtype is not None:
            data_dtype = required_label_dtype

        converted_data = convert_data(
            subset_content, data_format, data_order, data_dtype, device
        )
        data_dict[subset_name] = converted_data
        if not is_label:
            data_description[subset_name] = {
                "format": data_format,
                "order": data_order,
                "dtype": data_dtype,
                "samples": converted_data.shape[0],
            }
            if len(converted_data.shape) == 2 and converted_data.shape[1] > 1:
                data_description[subset_name]["features"] = converted_data.shape[1]

    return (
        tuple(data_dict[name] for name in ["x_train", "x_test", "y_train", "y_test"]),
        data_description,
    )
