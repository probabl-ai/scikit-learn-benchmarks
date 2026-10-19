"""KMeans matrix of `_kmeans.py` that times the Lloyd iterations.

The init picks random samples as centers (`init="random"`), so it takes
almost no time and the fit time is mostly Lloyd iterations.

Every fit runs up to 100 Lloyd iterations (`tol=0`, `max_iter=100`): with
the default `tol`, the iteration count varied up to 5x between repeats, and so
did the fit time. `tol=0` does not disable strict convergence (labels that
stop changing), which can still end the fits with many clusters earlier.
"""
from _kmeans import generate_kmeans_cases


def generate_cases() -> list[dict]:
    return generate_kmeans_cases({"init": "random", "tol": 0, "max_iter": 100})
