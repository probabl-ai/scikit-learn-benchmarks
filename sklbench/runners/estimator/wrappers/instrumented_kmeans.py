from __future__ import annotations

import contextlib as _contextlib
from dataclasses import dataclass as _dataclass
import time as _time

from sklearn.cluster import KMeans as _KMeans
from sklearn.cluster import _kmeans as _kmeans_module

__all__ = [
    "KMeans",
]


@_dataclass
class _KMeansTimings:
    init: float = 0.0
    lloyd: float = 0.0
    elkan: float = 0.0
    lloyd_iter: float = 0.0
    inertia: float = 0.0
    n_iter_total: int = 0


@_contextlib.contextmanager
def _instrument_kmeans(estimator):
    timings = _KMeansTimings()
    original_init_centroids = estimator._init_centroids
    patched = {
        name: getattr(_kmeans_module, name)
        for name in [
            "_kmeans_single_lloyd",
            "_kmeans_single_elkan",
            "lloyd_iter_chunked_dense",
            "lloyd_iter_chunked_sparse",
            "_inertia_dense",
            "_inertia_sparse",
        ]
    }

    def timed(func, field):
        def wrapper(*args, **kwargs):
            t0 = _time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dt = _time.perf_counter() - t0
                setattr(timings, field, getattr(timings, field) + dt)

        return wrapper

    def timed_single_run(func, field):
        timed_func = timed(func, field)

        def wrapper(*args, **kwargs):
            result = timed_func(*args, **kwargs)
            timings.n_iter_total += result[-1]
            return result

        return wrapper

    # Module globals are looked up at call time by `KMeans.fit` and
    # `_kmeans_single_*`, so swapping them is enough. The single-run functions
    # are the threadpoolctl-decorated ones, so `lloyd_time_` includes the cost
    # of entering/leaving that BLAS limit context.
    estimator._init_centroids = timed(original_init_centroids, "init")
    _kmeans_module._kmeans_single_lloyd = timed_single_run(
        patched["_kmeans_single_lloyd"], "lloyd"
    )
    _kmeans_module._kmeans_single_elkan = timed_single_run(
        patched["_kmeans_single_elkan"], "elkan"
    )
    for name in ["lloyd_iter_chunked_dense", "lloyd_iter_chunked_sparse"]:
        setattr(_kmeans_module, name, timed(patched[name], "lloyd_iter"))
    for name in ["_inertia_dense", "_inertia_sparse"]:
        setattr(_kmeans_module, name, timed(patched[name], "inertia"))
    try:
        yield timings
    finally:
        del estimator._init_centroids
        for name, func in patched.items():
            setattr(_kmeans_module, name, func)


class KMeans(_KMeans):
    def fit(self, X, y=None, sample_weight=None):
        t0 = _time.perf_counter()
        with _instrument_kmeans(self) as timings:
            result = super().fit(X, y=y, sample_weight=sample_weight)
        total = _time.perf_counter() - t0

        # Summed over all `n_init` runs. `other_time_` is input validation,
        # the `copy_x` copy, mean centering and `row_norms`.
        self.total_time_ = total
        self.init_time_ = timings.init
        self.lloyd_time_ = timings.lloyd
        self.elkan_time_ = timings.elkan
        # Lloyd only: the E+M step kernel (including the final E-step re-run
        # when convergence is not strict). The rest of `lloyd_time_` is
        # `inertia_time_` plus convergence checks and threadpoolctl overhead.
        self.lloyd_iter_time_ = timings.lloyd_iter
        self.inertia_time_ = timings.inertia
        self.other_time_ = total - timings.init - timings.lloyd - timings.elkan
        self.n_iter_total_ = timings.n_iter_total
        self.kmeans_n_threads_ = self._n_threads
        return result
