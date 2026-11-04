"""Lasso-like coordinate descent loop calling OpenBLAS ddot/daxpy via ctypes.

Per epoch and feature j: R += w X[:, j]; t = X[:, j] . R; R -= w X[:, j], the
same calls as sklearn's enet_coordinate_descent. Prints ms per 20 epochs.
"""

import ctypes
import os
import sys
import time

import numpy as np

lib = ctypes.CDLL(os.environ["LIB"])
n, d = int(sys.argv[1]), int(sys.argv[2])
offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
dot, axpy = lib.cblas_ddot, lib.cblas_daxpy
dot.restype = ctypes.c_double
dot.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
axpy.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
rng = np.random.default_rng(0)
X = np.asfortranarray(rng.standard_normal((n, d)))
Rbuf = rng.standard_normal(n + offset)
R = Rbuf[offset:]
cols = [X.ctypes.data + j * n * 8 for j in range(d)]
r = R.ctypes.data


def run():
    for _ in range(20):
        for c in cols:
            axpy(n, 0.001, c, 1, r, 1)
            dot(n, c, 1, r, 1)
            axpy(n, -0.001, c, 1, r, 1)


run()
times = []
for _ in range(10):
    t = time.perf_counter()
    run()
    times.append(time.perf_counter() - t)
print(f"{min(times) * 1e3:.1f} ms")
