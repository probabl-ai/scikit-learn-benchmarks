"""One timed run of a scikit-learn workload using prange, in the current build. Prints one JSON line.

usage: other_case.py CASE THREADS
"""
import json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits

case, threads = sys.argv[1], int(sys.argv[2])
rng = np.random.RandomState(0)
if case.startswith("kmeans"):
    from sklearn.cluster import KMeans
    algo, n, d, k = {"kmeans_lloyd": ("lloyd", 200_000, 50, 100), "kmeans_elkan": ("elkan", 200_000, 50, 100),
                     "kmeans_small": ("lloyd", 2_000, 10, 8)}[case]
    X = rng.standard_normal((n, d))
    init = X[:k].copy()
    est = KMeans(n_clusters=k, algorithm=algo, init=init, n_init=1, max_iter=30, tol=0, random_state=0)
    fn = lambda: est.fit(X)
    check = lambda: float(est.inertia_)
elif case.startswith("argkmin"):
    from sklearn.neighbors import NearestNeighbors
    n, d = {"argkmin": (40_000, 30), "argkmin_small": (1_000, 10)}[case]
    X = rng.standard_normal((n, d))
    nn = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(X)
    fn = lambda: nn.kneighbors(X[: n // 4])
    check = lambda: float(fn()[0].sum())
elif case == "radius":
    from sklearn.neighbors import NearestNeighbors
    X = rng.standard_normal((30_000, 10))
    nn = NearestNeighbors(radius=2.0, algorithm="brute").fit(X)
    fn = lambda: nn.radius_neighbors(X[:5000], return_distance=True)
    check = lambda: float(sum(d.sum() for d in fn()[0]))
elif case == "tsne":
    from sklearn.manifold import TSNE
    X = rng.standard_normal((4_000, 20))
    est = TSNE(max_iter=250, init="random", random_state=0, method="barnes_hut")
    fn = lambda: est.fit(X)
    check = lambda: float(est.kl_divergence_)
elif case == "hgb_predict":
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = rng.standard_normal((100_000, 20))
    y = X[:, 0] * X[:, 1] + rng.standard_normal(100_000)
    with threadpool_limits(limits=threads, user_api="openmp"):
        est = HistGradientBoostingRegressor(max_iter=50, random_state=0).fit(X, y)
    fn = lambda: [est.predict(X[:5000]) for _ in range(40)]
    check = lambda: float(est.predict(X[:5000]).sum())
else:
    raise SystemExit(f"unknown case {case}")

with threadpool_limits(limits=threads, user_api="openmp"), threadpool_limits(limits=1, user_api="blas"):
    t0 = time.perf_counter()
    fn()
    t = time.perf_counter() - t0
    value = check()
print(json.dumps({"case": case, "threads": threads, "time_s": t, "check": value}))
