"""Time a few scikit-learn and scipy calls that go through the patched routines.

Run once per BLAS variant (OpenBLAS swapped with LD_PRELOAD), e.g.

    LD_PRELOAD=builds/tc/lib/libopenblas.so.0 OPENBLAS_NUM_THREADS=86 \
        python scripts/sklearn_e2e.py --label tc

Prints one CSV row per case: label,threads,case,best_s,median_s.
"""

import argparse
import os
import time
import warnings

import numpy as np
import scipy.linalg
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso

warnings.simplefilter("ignore", ConvergenceWarning)


def timeit(f, min_time=1.0, min_reps=3):
    f()
    times = []
    start = time.perf_counter()
    while len(times) < min_reps or time.perf_counter() - start < min_time:
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    times.sort()
    return times[0], times[len(times) // 2]


def cases():
    rng = np.random.default_rng(0)
    for n, d in [(20000, 100), (100000, 100), (1000000, 20)]:
        X = np.asfortranarray(rng.standard_normal((n, d)))
        y = X[:, :10].sum(axis=1) + rng.standard_normal(n)
        # tol=0 and a fixed max_iter so both variants do the same epochs.
        yield f"Lasso {n}x{d}", lambda X=X, y=y: Lasso(alpha=0.01, tol=0, max_iter=20).fit(X, y)
        yield f"ElasticNet {n}x{d}", lambda X=X, y=y: ElasticNet(
            alpha=0.01, l1_ratio=0.5, tol=0, max_iter=20).fit(X, y)
    for n in [1000, 2000, 4000]:
        A = rng.standard_normal((n, n))
        b = rng.standard_normal(n)
        yield f"scipy.linalg.solve {n}", lambda A=A, b=b: scipy.linalg.solve(A, b)
        S = A + A.T
        yield f"numpy.linalg.eigh {n}", lambda S=S: np.linalg.eigh(S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--only", default="")
    args = p.parse_args()
    threads = os.environ.get("OPENBLAS_NUM_THREADS", "")
    for name, f in cases():
        if args.only and args.only not in name:
            continue
        best, med = timeit(f)
        print(f"{args.label},{threads},{name},{best:.6g},{med:.6g}", flush=True)


if __name__ == "__main__":
    main()
