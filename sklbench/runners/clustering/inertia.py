"""The quality metric of the k-means Pareto benchmark: the mean squared distance of
every sample to its nearest center, computed from the centers each implementation
returns rather than from its own ``inertia_`` (FAISS, for example, reports it on its
training subsample).

The nearest center comes from a float32 GEMM (``|c|^2 - 2 x.c``), then the squared
distance to that center is recomputed and summed in float64. The float32 search can
only pick a wrong center when two centers are within its rounding error (about
``1e-7 * |x|^2``) of each other, and the distance is then off by at most that much,
so the result ranks runs exactly. ``exact=True`` searches in float64 too (for tests).

The same code runs on NumPy, CuPy and PyTorch arrays, so GPU cases are scored on
the device that already holds the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InertiaResult:
    inertia: float  # mean squared distance to the nearest center
    n_empty: int  # centers no sample is closest to, non-finite ones included
    n_nonfinite: int  # centers with NaN or infinite coordinates, left out


class _Numpy:
    xp = np

    def asarray(self, a, float64=False):
        return np.asarray(a, dtype=np.float64 if float64 else np.float32)

    def to_float64(self, a):
        return a.astype(np.float64)

    def argmin(self, a):
        return a.argmin(axis=1)

    def scalar(self, a):
        return float(a)

    def bincount(self, labels, k):
        return np.bincount(labels, minlength=k)

    def to_numpy(self, a):
        return np.asarray(a)


class _Cupy(_Numpy):
    def __init__(self):
        import cupy

        self.xp = cupy

    def asarray(self, a, float64=False):
        return self.xp.asarray(a, dtype=self.xp.float64 if float64 else self.xp.float32)

    def bincount(self, labels, k):
        return self.xp.bincount(labels, minlength=k)

    def to_numpy(self, a):
        return self.xp.asnumpy(a)


class _Torch:
    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = device

    def asarray(self, a, float64=False):
        dtype = self.torch.float64 if float64 else self.torch.float32
        if not isinstance(a, self.torch.Tensor):
            a = self.torch.as_tensor(np.asarray(a))
        return a.to(device=self.device, dtype=dtype)

    def to_float64(self, a):
        return a.to(self.torch.float64)

    def argmin(self, a):
        return a.argmin(dim=1)

    def scalar(self, a):
        return float(a.item())

    def bincount(self, labels, k):
        return self.torch.bincount(labels, minlength=k)

    def to_numpy(self, a):
        return a.detach().cpu().numpy()


def _backend(X):
    module = type(X).__module__
    if module.startswith("cupy"):
        return _Cupy()
    if module.startswith("torch"):
        return _Torch(X.device)
    return _Numpy()


def to_numpy(a) -> np.ndarray:
    module = type(a).__module__
    if module.startswith("cupy"):
        import cupy

        return cupy.asnumpy(a)
    if module.startswith("torch"):
        return a.detach().float().cpu().numpy()
    return np.asarray(a)


def evaluate_inertia(X, centers, *, exact=False, chunk_bytes=256 << 20) -> InertiaResult:
    """Mean squared distance of the rows of ``X`` to their nearest row of ``centers``.

    ``X`` is a NumPy, CuPy or PyTorch array of shape (n, d), ``centers`` any array of
    shape (k, d). Centers with non-finite coordinates are left out (and counted)
    instead of making the result NaN.
    """
    be = _backend(X)
    centers = to_numpy(centers)
    if centers.ndim != 2 or centers.shape[1] != X.shape[1]:
        raise ValueError(f"centers of shape {centers.shape} don't match X of shape {X.shape}")
    finite = np.isfinite(centers).all(axis=1)
    n_nonfinite = int((~finite).sum())
    centers = centers[finite].astype(np.float64)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    k = centers.shape[0]

    C64 = be.asarray(centers, float64=True)
    C_search = C64 if exact else be.asarray(centers.astype(np.float32))
    C_norm = (C_search * C_search).sum(1)
    C_search_T = C_search.T

    # Bounds the float64 copy of a chunk and its (chunk, k) score matrix.
    chunk = max(1024, min(n_samples, chunk_bytes // (8 * max(n_features, k))))
    total = 0.0
    counts = None
    for start in range(0, n_samples, chunk):
        Xc = X[start : start + chunk]
        Xs = be.to_float64(Xc) if exact else be.asarray(Xc)
        labels = be.argmin(C_norm[None, :] - 2 * (Xs @ C_search_T))
        diff = be.to_float64(Xc) - C64[labels]
        total += be.scalar((diff * diff).sum())
        c = be.bincount(labels, k)
        counts = c if counts is None else counts + c
    n_empty = int((be.to_numpy(counts) == 0).sum()) + n_nonfinite
    return InertiaResult(inertia=total / n_samples, n_empty=n_empty, n_nonfinite=n_nonfinite)
