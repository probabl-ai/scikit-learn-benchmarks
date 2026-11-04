"""Fits on 4 threads with every threaded path of the HGB fit forced, for TSan."""
import os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as gb

T = 4
gb.BaseHistGradientBoosting._get_heuristic_optimal_n_threads = staticmethod(lambda *a, **k: T)
gb._openmp_uses_active_wait = lambda: True  # threaded split_indices from 500 samples per thread
rng = np.random.RandomState(0)
n, p = 20_000, 12
X = rng.randn(n, p)
X[rng.rand(n, p) < 0.05] = np.nan
X[:, 0] = rng.randint(0, 8, n)
y = np.nansum(X[:, 1:4], axis=1) + rng.randn(n)
y_clf = np.digitize(y, [-1, 1])
for Est, target, kw in [
    (R, y, dict(max_leaf_nodes=31)),
    (R, y, dict(categorical_features=[0], max_features=0.5, max_leaf_nodes=31)),
    (R, y, dict(monotonic_cst=[0, 1, -1] + [0] * (p - 3), max_leaf_nodes=31)),
    (R, y, dict(interaction_cst=[[1, 2, 3], [4, 5, 6, 7]], max_leaf_nodes=31)),
    (C, y_clf, dict(max_depth=4)),  # non constant hessians, children_are_leaves
]:
    m = Est(max_iter=5, random_state=0, early_stopping=False, **kw).fit(X, target)
    print(Est.__name__, kw, "score", round(m.score(X, target), 4), flush=True)
