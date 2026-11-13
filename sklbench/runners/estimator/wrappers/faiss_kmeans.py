"""FAISS k-means (Lloyd) and e-means behind a scikit-learn-like ``fit``.

e-means (``faiss.contrib.e_means``, arXiv 2611.09501) is mini-batch k-means with
exponential moving average updates of the centers, and random init. ``n_epochs`` (passes
over the data, fractional allowed) is its budget. Batch size and step size follow the
paper's scaling laws (``batch_law="auto"``, or ``"quality"`` for ~8x smaller batches)
unless ``batch_size``/``alpha0`` are set.

FAISS is imported in ``fit``, so that other environments can import this module.
"""

from __future__ import annotations

import os

from sklearn.base import BaseEstimator


def _threads() -> int:
    return int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())


class Kmeans(BaseEstimator):
    """``faiss.Kmeans``: Lloyd with random init and a fixed number of iterations.

    FAISS subsamples to ``max_points_per_centroid * n_clusters`` rows by default (256,
    so 65,536 rows at k=256). Here ``None`` trains on all the rows, and a number
    subsamples on purpose. ``nredo`` is the number of inits.

    With ``gpu=True``, the training loop still runs on the host: only the nearest
    center search runs on the GPU, so the fit time includes the transfers of every
    iteration. That's how ``faiss.Kmeans(gpu=True)`` works.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        niter=25,
        nredo=1,
        max_points_per_centroid=None,
        gpu=False,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.niter = niter
        self.nredo = nredo
        self.max_points_per_centroid = max_points_per_centroid
        self.gpu = gpu
        self.random_state = random_state

    def fit(self, X, y=None):
        import faiss

        if self.gpu and faiss.get_num_gpus() < 1:
            raise RuntimeError("FAISS sees no GPU")
        n_samples, n_features = X.shape
        max_points = self.max_points_per_centroid
        if max_points is None:
            max_points = n_samples // self.n_clusters + 1
        km = faiss.Kmeans(
            n_features,
            self.n_clusters,
            niter=self.niter,
            nredo=self.nredo,
            seed=self.random_state or 0,
            gpu=self.gpu,
            max_points_per_centroid=max_points,
            min_points_per_centroid=1,
            verbose=False,
        )
        km.train(X)
        n_train = min(n_samples, max_points * self.n_clusters)
        self.cluster_centers_ = km.centroids
        self.n_iter_ = self.niter
        self.n_iter_total_ = self.niter * self.nredo
        self.n_epochs_ = self.n_iter_total_ * n_train / n_samples
        self.n_train_samples_ = n_train
        # FAISS's objective is measured on its training rows only.
        if len(km.obj):
            self.inertia_ = float(km.obj[-1]) * n_samples / n_train
        return self


class _Emeans(BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        *,
        n_epochs=5,
        batch_law="auto",
        batch_size=None,
        alpha0=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.random_state = random_state

    def _kwargs(self) -> dict:
        kwargs = {
            "nepochs": self.n_epochs,
            "batch_law": self.batch_law,
            "seed": self.random_state or 0,
        }
        for name in ("batch_size", "alpha0"):
            if getattr(self, name) is not None:
                kwargs[name] = getattr(self, name)
        return kwargs

    def _train(self, em, X):
        centers, _ = em.train(X)
        self.cluster_centers_ = centers
        recipe = em.recipe or {}
        steps = recipe.get("steps_per_pass")
        self.n_iter_ = int(round(steps * self.n_epochs)) if steps else None
        self.n_epochs_ = float(self.n_epochs)
        # The recipe the scaling laws picked.
        for name in ("batch_size", "alpha0", "steps_per_pass", "assignment_backend"):
            value = recipe.get(name)
            if value is not None:
                setattr(self, f"{name}_", value if isinstance(value, (int, float)) else str(value))
        self.n_splits_ = int(getattr(em, "n_splits", 0))
        return self


class NumpyEmeans(_Emeans):
    """``faiss.contrib.e_means.NumpyEmeans`` on host data (``faiss.knn`` in float32)."""

    def fit(self, X, y=None):
        from faiss.contrib.e_means import NumpyEmeans
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=_threads()):
            return self._train(NumpyEmeans(X.shape[1], self.n_clusters, **self._kwargs()), X)


class FastNumpyEmeans(_Emeans):
    """``NumpyEmeans`` with its accumulation step in an OpenMP Cython kernel (see
    `faiss_emeans_cpu`): the same algorithm and recipe, without ``np.add.at``, which
    takes ~75% of an epoch upstream."""

    def fit(self, X, y=None):
        from threadpoolctl import threadpool_limits

        from .faiss_emeans_cpu import FastNumpyEmeans

        em = FastNumpyEmeans(
            X.shape[1], self.n_clusters, n_threads=_threads(), **self._kwargs()
        )
        # The parallelism is in the kernel's OpenMP loops, so its sgemm calls must
        # run on one thread.
        with threadpool_limits(limits=1, user_api="blas"):
            return self._train(em, X)


class RustEmeans(_Emeans):
    """``Emeans`` of the `emeans <https://pypi.org/project/emeans/>`_ package: a Rust
    port of ``NumpyEmeans``, threaded with rayon."""

    def fit(self, X, y=None):
        # rayon sizes its global pool once, at the first call.
        os.environ.setdefault("RAYON_NUM_THREADS", str(_threads()))
        from emeans import Emeans

        return self._train(Emeans(X.shape[1], self.n_clusters, **self._kwargs()), X)


def _disable_tf32():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


class QuantizedEmeans(_Emeans):
    """``QuantizedEmeans(quantizer=None)`` on a CUDA float32 tensor: native FP32 with
    FAISS's Triton kernels."""

    def fit(self, X, y=None):
        from faiss.contrib.e_means import QuantizedEmeans

        _disable_tf32()
        em = QuantizedEmeans(X.shape[1], self.n_clusters, quantizer=None, **self._kwargs())
        return self._train(em, X)


class Emeans(_Emeans):
    """``faiss.contrib.e_means.Emeans`` on a CUDA float32 tensor. By default it turns
    the FP32 input into BF16 operands: reduced precision."""

    def fit(self, X, y=None):
        from faiss.contrib.e_means import Emeans

        _disable_tf32()
        return self._train(Emeans(X.shape[1], self.n_clusters, **self._kwargs()), X)
