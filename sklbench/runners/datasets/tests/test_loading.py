import numpy as np
import scipy.sparse as sp

from sklbench.runners.datasets.loading import load_data_from_cache, save_data_to_cache


def test_sparse_data_round_trips_through_cache(tmp_path):
    x = sp.random(20, 7, density=0.2, format="csr", random_state=0)
    y = np.arange(20)

    save_data_to_cache({"x": x, "y": y}, str(tmp_path), "sparse_set")
    loaded = load_data_from_cache(str(tmp_path), "sparse_set")

    assert sp.issparse(loaded["x"]) and loaded["x"].format == "csr"
    assert (loaded["x"] != x).nnz == 0
    np.testing.assert_array_equal(loaded["y"], y)
