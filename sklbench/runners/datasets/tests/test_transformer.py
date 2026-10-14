import numpy as np
import pytest

from sklbench.runners.datasets import transformer
from sklbench.runners.datasets.transformer import convert_data

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "data_dtype, requested_dtype, max_precision, expected_fallbacks",
    [
        (np.float64, None, "float32", {"float64"}),
        (np.float32, "float64", "float32", {"float64"}),
        (np.float32, None, "float32", set()),
        (np.float64, "float32", "float32", set()),
        (np.float64, None, "float64", set()),
    ],
)
def test_convert_data_reports_float64_fallback(
    monkeypatch, data_dtype, requested_dtype, max_precision, expected_fallbacks
):
    # Stands in for a device without float64 support (MPS).
    monkeypatch.setattr(
        transformer,
        "_torch_max_precision_float_dtype",
        lambda device: getattr(torch, max_precision),
    )
    fallbacks = set()

    converted = convert_data(
        np.zeros((3, 2), dtype=data_dtype),
        dformat="torch",
        dtype=requested_dtype,
        device="cpu",
        on_dtype_fallback=fallbacks.add,
    )

    assert fallbacks == expected_fallbacks
    if expected_fallbacks:
        assert converted.dtype == torch.float32
