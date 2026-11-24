"""e-means on the CPU with a compiled accumulation step.

``FastNumpyEmeans`` is FAISS's ``NumpyEmeans`` (faiss 1.16.0, ``faiss.contrib.e_means``)
with one method replaced: ``_accumulate_core``, which assigns a mini-batch to its nearest
centroids and adds it to the per-cluster sums and counts. Upstream does that with
``faiss.knn`` on a gathered copy of the batch, then ``np.bincount`` and
``np.add.at(sums, labels, rows)``. Here one OpenMP kernel (``_emeans_kernel.pyx``) does
it all. Everything else is FAISS's code, unchanged: the scaling laws for the batch size
and step size, the EMA update, the revival of unused clusters, the epoch order and the
closing means.

The method mirrors the upstream one (pending statistics across partial calls, captured
labels in the last epoch), and is pinned to faiss 1.16.0, whose private method it
overrides. Spherical mode and batches above 2**24 rows (float64 accumulation) are not
supported.
"""

from __future__ import annotations

import os

import numpy as np
from faiss.contrib.e_means import NumpyEmeans

try:
    from ._emeans_kernel import accumulate_step
except ImportError as exc:
    raise ImportError(
        "the e-means kernel is not built: run `pixi run -e faiss-cpu build-emeans-kernel`"
    ) from exc


def _n_threads():
    return int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())


class FastNumpyEmeans(NumpyEmeans):
    """``NumpyEmeans`` with the accumulation step in a compiled OpenMP kernel."""

    def __init__(self, *args, n_threads=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.spherical:
            raise NotImplementedError("spherical mode is not supported")
        self.n_threads = n_threads or _n_threads()

    def _chunk_rows(self, n):
        # Enough chunks to keep every thread busy on small batches (the "quality" law
        # uses ~8x smaller batches), large enough chunks for an efficient sgemm.
        return int(min(256, max(32, -(-n // (4 * self.n_threads)))))

    def _accumulate_core(self, x, index=None, labels_out=None):
        n = len(x) if index is None else len(index)
        pending_n = 0 if self._pending_stats is None else self._pending_stats["n"]
        target = self._local_batch_size()
        if target > 2**24:
            raise NotImplementedError("batches above 2**24 rows need float64 sums")
        if pending_n == 0:
            self._batch_stats.fill(0)
            self._batch_objective = np.float64(0)
            self._batch_centroids = self._readout()
            self._batch_accumulation = self._batch_stats
            self._batch_norms = np.einsum("ij,ij->i", self._batch_centroids,
                                          self._batch_centroids)
        accumulation = self._batch_accumulation
        centroids = np.ascontiguousarray(self._batch_centroids, dtype=np.float32)
        sums, counts = accumulation[:, :-1], accumulation[:, -1]
        captured = None if labels_out is None else labels_out[:n]
        if (captured is None and self.capture_assignments
                and self._epoch == len(self.recipe["epoch_rows"]) - 1):
            captured = np.empty(n, dtype="int64")
        if index is None:
            index = np.arange(n, dtype=np.int64)
        index = np.ascontiguousarray(index, dtype=np.int64)
        labels = captured if captured is not None else np.empty(n, dtype=np.int64)
        order = np.empty(n, dtype=np.int64)
        cluster_start = np.empty(self.k + 1, dtype=np.int64)
        objective = accumulate_step(
            np.ascontiguousarray(x, dtype=np.float32), index, centroids,
            np.ascontiguousarray(self._batch_norms, dtype=np.float32), sums, counts,
            labels, order, cluster_start, self._chunk_rows(n), self.n_threads)
        self._batch_objective = self._batch_objective + objective
        total_n = pending_n + n
        complete = total_n == target or total_n == 0
        stats = {
            "sums": sums if complete else None,
            "counts": counts if complete else None,
            "objective": self._batch_objective,
            "n": total_n,
        }
        if captured is not None:
            stats["labels"] = captured
        return stats
