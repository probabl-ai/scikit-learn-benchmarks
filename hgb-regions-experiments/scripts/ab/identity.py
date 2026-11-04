"""Hash of the trees of several fits with a forced team size, to compare builds."""
import hashlib, json, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as gb
out = {}
rng = np.random.RandomState(0)
n, p = 30_000, 40
X = rng.randn(n, p); X[rng.rand(n, p) < 0.05] = np.nan; X[:, 0] = rng.randint(0, 12, n)
y_reg = np.nansum(X[:, :5], axis=1) + rng.randn(n); y_clf = np.digitize(y_reg, [-1, 1])
cases = {
    "reg_missing_cat": (R, y_reg, dict(categorical_features=[0], max_leaf_nodes=63)),
    "reg_monotonic": (R, y_reg, dict(monotonic_cst=[0, 1, -1] + [0] * (p - 3), max_leaf_nodes=31)),
    "reg_interaction": (R, y_reg, dict(interaction_cst=[[1, 2, 3], [4, 5, 6, 7]], max_leaf_nodes=31)),
    "reg_maxfeat": (R, y_reg, dict(max_features=0.3, max_leaf_nodes=127, min_samples_leaf=5)),
    "reg_depth": (R, y_reg, dict(max_depth=3, max_leaf_nodes=None)),
    "clf3_cat_maxfeat": (C, y_clf, dict(categorical_features=[0], max_features=0.5, max_depth=6)),
}
for T in (1, 8):
    if hasattr(gb.BaseHistGradientBoosting, "_get_heuristic_optimal_n_threads"):
        gb.BaseHistGradientBoosting._get_heuristic_optimal_n_threads = staticmethod(lambda *a, **k: T)
    else:
        gb._openmp_effective_n_threads = lambda n_threads=None: T
    for name, (Est, y, kw) in cases.items():
        m = Est(max_iter=15, random_state=0, early_stopping=False, **kw).fit(X, y)
        h = hashlib.sha256()
        for preds in m._predictors:
            for pr in preds:
                h.update(pr.nodes.tobytes())
        h.update((m.predict_proba(X) if Est is C else m.predict(X)).tobytes())
        out[f"{name}/T{T}"] = h.hexdigest()[:16]
json.dump(out, open(sys.argv[1], "w"), indent=0, sort_keys=True)
print("written", sys.argv[1], len(out))
