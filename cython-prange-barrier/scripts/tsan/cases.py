"""Small runs of the scikit-learn code using prange, on 4 threads, for TSan. Prints one value per case."""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as gb
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, RadiusNeighborsClassifier

T = 4
if hasattr(gb.BaseHistGradientBoosting, "_get_heuristic_optimal_n_threads"):
    # Force every threaded path of the HGB fit, on the branch with the guards.
    gb.BaseHistGradientBoosting._get_heuristic_optimal_n_threads = staticmethod(lambda *a, **k: T)
if hasattr(gb, "_openmp_uses_active_wait"):
    gb._openmp_uses_active_wait = lambda: True

rng = np.random.RandomState(0)
n, p = 20_000, 12
X = rng.randn(n, p)
y = X[:, 1:4].sum(axis=1) + rng.randn(n)
y_clf = np.digitize(y, [-1, 1])
X_nan = X.copy()
X_nan[rng.rand(n, p) < 0.05] = np.nan
X_nan[:, 0] = rng.randint(0, 8, n)


def report(name, value):
    print(f"{name}: {value:.10g}", flush=True)


for name, kw in [
    ("hgb_plain", dict(max_leaf_nodes=31)),
    ("hgb_categorical", dict(categorical_features=[0], max_features=0.5, max_leaf_nodes=31)),
    ("hgb_monotonic", dict(monotonic_cst=[0, 1, -1] + [0] * (p - 3), max_leaf_nodes=31)),
    ("hgb_interaction", dict(interaction_cst=[[1, 2, 3], [4, 5, 6, 7]], max_leaf_nodes=31)),
]:
    m = R(max_iter=5, random_state=0, early_stopping=False, **kw).fit(X_nan, y)
    report(name, m.score(X_nan, y))
m = C(max_iter=5, max_depth=4, random_state=0, early_stopping=False).fit(X_nan, y_clf)
report("hgb_classifier", m.score(X_nan, y_clf))

for algo in ("lloyd", "elkan"):
    km = KMeans(n_clusters=8, algorithm=algo, init=X[:8].copy(), n_init=1, max_iter=5, tol=0).fit(X)
    report(f"kmeans_{algo}", km.inertia_)
km = MiniBatchKMeans(n_clusters=8, init=X[:8].copy(), n_init=1, max_iter=3, batch_size=2048, random_state=0).fit(X)
report("kmeans_minibatch", km.inertia_)

X_small = X[:4000]
y_small = y_clf[:4000]
nn = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(X_small)
report("argkmin", nn.kneighbors(X_small[:1000])[0].sum())
report("argkmin_cdist", pairwise_distances_argmin(X_small[:1000], X_small[1000:2000]).sum())
nn = NearestNeighbors(radius=2.0, algorithm="brute").fit(X_small)
report("radius", sum(d.sum() for d in nn.radius_neighbors(X_small[:1000])[0]))
report("argkmin_classmode", KNeighborsClassifier(algorithm="brute").fit(X_small, y_small).predict_proba(X_small[:1000]).sum())
report("radius_classmode", RadiusNeighborsClassifier(radius=4.0, algorithm="brute", outlier_label="most_frequent").fit(X_small, y_small).predict_proba(X_small[:1000]).sum())

est = TSNE(max_iter=250, init="random", random_state=0, method="barnes_hut", perplexity=10).fit(X[:600])
report("tsne", est.kl_divergence_)
report("loss_binomial", LogisticRegression(max_iter=5).fit(X, y_clf > 0).score(X, y_clf > 0))
report("loss_multinomial", LogisticRegression(max_iter=5).fit(X, y_clf).score(X, y_clf))
report("loss_poisson", PoissonRegressor(max_iter=5).fit(X, np.abs(y)).score(X, np.abs(y)))
print("done", flush=True)
