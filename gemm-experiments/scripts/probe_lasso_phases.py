"""Split Lasso 1000000x20 fit time: full fit, fit with max_iter=1, and numpy-only steps."""

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

warnings.simplefilter("ignore", ConvergenceWarning)
rng = np.random.default_rng(0)
X = np.asfortranarray(rng.standard_normal((1000000, 20)))
y = X[:, :10].sum(axis=1) + rng.standard_normal(1000000)


def best(f, reps=10):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


full = best(lambda: Lasso(alpha=0.01, tol=0, max_iter=20).fit(X, y))
one = best(lambda: Lasso(alpha=0.01, tol=0, max_iter=1).fit(X, y))
copy = best(lambda: X - X.mean(axis=0))
print(f"full {full:.1f} ms, max_iter=1 {one:.1f} ms, X - mean {copy:.1f} ms")
