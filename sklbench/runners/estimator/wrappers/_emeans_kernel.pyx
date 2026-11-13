# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""One e-means accumulation step on the CPU, with OpenMP threads.

For the rows ``X[index]`` of one mini-batch, find each row's nearest centroid, then add
the rows to the per-cluster sums and counts, and return the summed squared distances.

1. Assignment, in parallel over chunks of rows: each thread gathers its chunk of rows
   into a private buffer (no separate gather copy of the batch), computes the dot products
   with all the centroids by one single-threaded BLAS ``sgemm``, and takes the argmin of
   ``|c|^2 - 2 x.c``. The squared distance ``|x|^2 + min`` is clamped at 0.
2. A counting sort of the batch's rows by label (serial, O(batch)).
3. Accumulation, in parallel over clusters: each thread sums the rows of its clusters, in
   float64, into the float32 sums. There is no ``np.add.at``, no atomics and no per-thread
   k x d reduction buffer, so the cost does not grow with k x d x threads.

BLAS must be single-threaded while this runs (``threadpool_limits(1, "blas")``): the
parallelism is in the OpenMP loops, as in scikit-learn's Lloyd kernels.
"""

from cython.parallel cimport prange, parallel
from libc.stdlib cimport malloc, free
from libc.float cimport FLT_MAX
from scipy.linalg.cython_blas cimport sgemm

import numpy as np


def accumulate_step(
    const float[:, ::1] X,
    const long long[::1] index,
    const float[:, ::1] centers,
    const float[::1] center_norms,
    float[:, :] sums,
    float[:] counts,
    long long[::1] labels,
    long long[::1] order,
    long long[::1] cluster_start,
    int chunk_rows,
    int n_threads,
):
    """Accumulate the rows ``X[index]`` into ``sums``/``counts``; return the objective.

    ``labels`` (len(index)) receives each row's nearest centroid. ``order`` (len(index))
    and ``cluster_start`` (k + 1) are scratch arrays. ``sums`` and ``counts`` may be
    strided views (e-means keeps them in one (k, d + 1) table) and are added to.
    """
    cdef Py_ssize_t n = index.shape[0]
    cdef int d = <int>X.shape[1]
    cdef int k = <int>centers.shape[0]
    cdef Py_ssize_t n_chunks = (n + chunk_rows - 1) // chunk_rows
    cdef Py_ssize_t c, i, r, start, stop, m, j, best_j, o, f
    cdef float *buf
    cdef float *scores
    cdef float best, value, xnorm, dist
    cdef double objective = 0
    cdef double *acc
    cdef long long row
    cdef char transa = b'T'
    cdef char transb = b'N'
    cdef float alpha = 1.0
    cdef float beta = 0.0
    cdef int mm

    if n == 0:
        return 0.0

    # 1. assignment, in parallel over chunks of rows
    with nogil, parallel(num_threads=n_threads):
        buf = <float *>malloc(chunk_rows * d * sizeof(float))
        scores = <float *>malloc(chunk_rows * k * sizeof(float))
        for c in prange(n_chunks, schedule='static'):
            start = c * chunk_rows
            stop = start + chunk_rows
            if stop > n:
                stop = n
            m = stop - start
            for r in range(m):
                row = index[start + r]
                for f in range(d):
                    buf[r * d + f] = X[row, f]
            # scores (m x k, row-major) = buf (m x d) . centers^T, as column-major
            # scores^T (k x m) = centers (k x d) . buf^T
            mm = <int>m
            sgemm(&transa, &transb, &k, &mm, &d, &alpha, <float *>&centers[0, 0], &d,
                  buf, &d, &beta, scores, &k)
            for r in range(m):
                best = FLT_MAX
                best_j = 0
                for j in range(k):
                    value = center_norms[j] - 2 * scores[r * k + j]
                    if value < best:
                        best = value
                        best_j = j
                xnorm = 0
                for f in range(d):
                    xnorm = xnorm + buf[r * d + f] * buf[r * d + f]
                dist = xnorm + best
                if dist < 0:
                    dist = 0
                labels[start + r] = best_j
                objective += dist
        free(buf)
        free(scores)

    # 2. counting sort of the rows by label
    cdef long long[::1] fill = np.zeros(k, dtype=np.int64)
    for j in range(k + 1):
        cluster_start[j] = 0
    for i in range(n):
        cluster_start[labels[i] + 1] += 1
    for j in range(k):
        cluster_start[j + 1] += cluster_start[j]
    for i in range(n):
        j = labels[i]
        order[cluster_start[j] + fill[j]] = index[i]
        fill[j] += 1

    # 3. accumulation, in parallel over clusters
    with nogil, parallel(num_threads=n_threads):
        acc = <double *>malloc(d * sizeof(double))
        for j in prange(k, schedule='dynamic'):
            if cluster_start[j + 1] > cluster_start[j]:
                for f in range(d):
                    acc[f] = 0
                for o in range(cluster_start[j], cluster_start[j + 1]):
                    row = order[o]
                    for f in range(d):
                        acc[f] = acc[f] + X[row, f]
                for f in range(d):
                    sums[j, f] = sums[j, f] + <float>acc[f]
                counts[j] = counts[j] + <float>(cluster_start[j + 1] - cluster_start[j])
        free(acc)

    return objective
