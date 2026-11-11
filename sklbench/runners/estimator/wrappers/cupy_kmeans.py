"""The CuPy k-means of ogrisel/cursor-sandbox#22 behind a scikit-learn-like ``fit``.

Its ``kmeans_lloyd`` package is fetched at a pinned commit into ``.cache/`` on first
use. ``fused=False`` runs CuPy array operations only, ``fused=True`` CUDA kernels that
CuPy compiles at run time. ``init="k-means++"`` runs the branch's ``kmeans_cupy``
(greedy k-means++ then Lloyd, ``n_init`` times). ``init="random"`` picks ``n_clusters``
distinct random rows per init and runs the same Lloyd. ``tol`` follows scikit-learn
(relative to the mean feature variance).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

import numpy as np
from sklearn.base import BaseEstimator

from ....config.registry import REPO_ROOT

REPOSITORY = "https://github.com/cakedev0/cursor-sandbox.git"
COMMIT = "6d8464c05e58131e7dc6496b65a88a62bf6326ea"  # branch cupy-kmeans, journal 024
CHECKOUT = REPO_ROOT / ".cache" / "cursor-sandbox"
PACKAGE_DIR = CHECKOUT / "kmeans_lloyd" / "src" / "kmeans_lloyd"


def _fetch():
    if not (CHECKOUT / ".git").exists():
        CHECKOUT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(CHECKOUT)], check=True)
        subprocess.run(["git", "-C", str(CHECKOUT), "remote", "add", "origin", REPOSITORY], check=True)
    head = subprocess.run(
        ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if head != COMMIT:
        git = ["git", "-C", str(CHECKOUT)]
        subprocess.run([*git, "fetch", "-q", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run([*git, "checkout", "-q", COMMIT], check=True)


def _cupy_lloyd():
    _fetch()
    # The package's __init__ imports its array API code, which needs the standalone
    # array-api-compat. The CuPy modules only need NumPy and CuPy, so they are
    # imported under an empty parent package.
    if "kmeans_lloyd" not in sys.modules:
        package = types.ModuleType("kmeans_lloyd")
        package.__path__ = [str(PACKAGE_DIR)]
        sys.modules["kmeans_lloyd"] = package
    return importlib.import_module("kmeans_lloyd.cupy_lloyd")


class KMeans(BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=1,
        max_iter=300,
        tol=1e-4,
        fused=True,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.fused = fused
        self.random_state = random_state

    def fit(self, X, y=None):
        mod = _cupy_lloyd()
        if self.init == "k-means++":
            best, runs = mod.kmeans_cupy(
                X,
                self.n_clusters,
                fused=self.fused,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
        elif self.init == "random":
            best, runs = self._fit_random(mod, X)
        else:
            raise ValueError(f"unsupported init {self.init!r}")
        centers, _, inertia, n_iter = best
        self.cluster_centers_ = centers
        self.inertia_ = float(inertia)
        self.n_iter_ = int(n_iter)
        self.n_iter_total_ = int(sum(run[1] for run in runs))
        return self

    def _fit_random(self, mod, X):
        import cupy as cp

        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        augmented = None
        if not self.fused:
            augmented = cp.concatenate([X, cp.ones((n, 1), dtype=X.dtype)], axis=1)
        best, runs = None, []
        for _ in range(self.n_init):
            centers = X[cp.asarray(np.sort(rng.choice(n, self.n_clusters, replace=False)))]
            if self.fused:
                result = mod.lloyd_cupy_fused(X, centers, max_iter=self.max_iter, tol=self.tol)
            else:
                result = mod.lloyd_cupy(
                    X, centers, max_iter=self.max_iter, tol=self.tol, augmented=augmented
                )
            runs.append((result[2], result[3]))
            if best is None or result[2] < best[2]:
                best = result
        return best, runs
