"""The quality metric of the k-means Pareto benchmark, and the e-means CPU kernel."""

import numpy as np
import pytest

from sklbench.runners.clustering.inertia import evaluate_inertia


def _brute_force_inertia(X, C):
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return float(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).min(axis=1).mean())


def _data(n=3000, d=17, k=40, offset=0.0, scale=1.0):
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(n, d)) * scale + offset).astype(np.float32)
    C = X[rng.choice(n, k, replace=False)] + rng.normal(scale=0.1 * scale, size=(k, d))
    return X, C.astype(np.float32)


@pytest.mark.parametrize("offset, scale", [(0.0, 1.0), (500.0, 30.0), (0.0, 1e-3)])
def test_inertia_matches_float64_brute_force(offset, scale):
    # A large offset (like SIFT, norms ~509) stresses the float32 search.
    X, C = _data(offset=offset, scale=scale)
    expected = _brute_force_inertia(X, C)
    assert evaluate_inertia(X, C).inertia == pytest.approx(expected, rel=1e-9)
    assert evaluate_inertia(X, C, chunk_bytes=1).inertia == pytest.approx(expected, rel=1e-9)


def test_inertia_skips_nonfinite_centers_and_counts_empty_ones():
    X, C = _data(n=1000, k=10)
    far = np.full((2, X.shape[1]), 1e6, dtype=np.float32)
    nan = np.full((1, X.shape[1]), np.nan, dtype=np.float32)
    result = evaluate_inertia(X, np.vstack([C, far, nan]))
    assert result.inertia == pytest.approx(_brute_force_inertia(X, C), rel=1e-9)
    assert result.n_nonfinite == 1
    assert result.n_empty >= 3


def test_inertia_on_sparse_matches_dense():
    sp = pytest.importorskip("scipy.sparse")
    X, C = _data(n=5000, k=30)
    X[np.abs(X) < 1] = 0  # ~70% zeros
    expected = evaluate_inertia(X, C)
    result = evaluate_inertia(sp.csr_matrix(X), C)
    assert result.inertia == pytest.approx(expected.inertia, rel=1e-9)
    assert result.n_empty == expected.n_empty


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_inertia_on_gpu_matches_numpy(backend):
    X, C = _data(offset=100.0, scale=10.0)
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            X_device = cp.asarray(X)
        except Exception:
            pytest.skip("no CUDA device")
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        X_device = torch.from_numpy(X).cuda()
    assert evaluate_inertia(X_device, C).inertia == pytest.approx(
        _brute_force_inertia(X, C), rel=1e-9
    )


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("d, k", [(3, 7), (200, 256)])
def test_emeans_kernel_step_matches_numpy(d, k, n_threads):
    kernel = pytest.importorskip(
        "sklbench.runners.estimator.wrappers._emeans_kernel",
        reason="run `pixi run -e faiss-cpu build-emeans-kernel`",
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, d)).astype(np.float32)
    C = X[rng.choice(5000, k, replace=False)] + 0.01
    index = rng.permutation(5000)[:1500].astype(np.int64)
    table = np.zeros((k, d + 1), dtype=np.float32)  # e-means' strided layout
    sums, counts = table[:, :-1], table[:, -1]
    labels = np.empty(len(index), dtype=np.int64)
    objective = kernel.accumulate_step(
        X, index, C, (C * C).sum(1), sums, counts, labels,
        np.empty(len(index), dtype=np.int64), np.empty(k + 1, dtype=np.int64), 64, n_threads,
    )

    rows = X[index].astype(np.float64)
    d2 = ((rows[:, None, :] - C[None].astype(np.float64)) ** 2).sum(-1)
    expected_labels = d2.argmin(1)
    expected_sums = np.zeros(C.shape)
    np.add.at(expected_sums, expected_labels, rows)
    assert (labels == expected_labels).mean() > 0.999  # float32 ties aside
    if (labels == expected_labels).all():
        np.testing.assert_allclose(sums, expected_sums, rtol=1e-5, atol=1e-4)
        np.testing.assert_array_equal(counts, np.bincount(expected_labels, minlength=k))
    assert objective == pytest.approx(d2.min(1).sum(), rel=1e-4)
