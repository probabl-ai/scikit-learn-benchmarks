import os, itertools
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as gb
gb.BaseHistGradientBoosting._get_heuristic_optimal_n_threads = staticmethod(lambda *a, **k: 8)
from sklearn.utils._openmp_helpers import _openmp_uses_active_wait
print(sklearn.__file__, "active_wait:", _openmp_uses_active_wait())
rng = np.random.RandomState(0)
n, p = 30_000, 40
X = rng.randn(n, p)
X[rng.rand(n, p) < 0.05] = np.nan
X[:, 0] = rng.randint(0, 12, n)                     # categorical
y_reg = np.nansum(X[:, :5], axis=1) + rng.randn(n)
y_clf = np.digitize(y_reg, [-1, 1])                 # 3 classes

cases = {
    "reg_missing_cat": (R, y_reg, dict(categorical_features=[0], max_leaf_nodes=63)),
    "reg_monotonic": (R, y_reg, dict(monotonic_cst=[0, 1, -1] + [0] * (p - 3), max_leaf_nodes=31)),
    "reg_interaction": (R, y_reg, dict(interaction_cst=[[1, 2, 3], [4, 5, 6, 7]], max_leaf_nodes=31)),
    "reg_maxfeat": (R, y_reg, dict(max_features=0.3, max_leaf_nodes=127, min_samples_leaf=5)),
    "clf3_cat_maxfeat": (C, y_clf, dict(categorical_features=[0], max_features=0.5, max_depth=6)),
}
variants = [("pr", ""), ("abs", ""), ("none", ""), ("pr", "fuse"), ("abs", "fuse"), ("none", "fuse"), ("pr", "fuse2"), ("abs-tiny", "fuse2"), ("pr", "fuse3"), ("abs", "fuse3"), ("abs-tiny", "fuse3"),
            ("abs-tiny", "fuse")]
for name, (Est, y, kw) in cases.items():
    ref = None
    for guard, exp in variants:
        os.environ["SKLEARN_HGB_GUARD"] = guard.split("-")[0]
        os.environ["SKLEARN_HGB_EXP"] = exp
        # abs-tiny: thresholds of 1 force the threaded paths everywhere
        os.environ["SKLEARN_HGB_ABS_SAMPLES"] = "1" if guard == "abs-tiny" else "20000"
        os.environ["SKLEARN_HGB_ABS_BINS"] = "1" if guard == "abs-tiny" else "4000"
        m = Est(max_iter=15, random_state=0, early_stopping=False, **kw).fit(X, y)
        out = m.predict_proba(X) if Est is C else m.predict(X)
        sig = [(pr.nodes.tobytes()) for preds in m._predictors for pr in preds]
        if ref is None:
            ref = (out, sig)
            status = "ref"
        else:
            status = "IDENTICAL" if (np.array_equal(out, ref[0]) and sig == ref[1]) else "DIFFERENT"
        print(f"{name:18s} guard={guard:8s} exp={exp or '-':5s} {status}")
