"""KMeans matrix of `_kmeans.py` that times both phases of a fit: the
k-means++ init and the Lloyd iterations.

Every fit runs up to 100 Lloyd iterations (`tol=0`, `max_iter=100`), as in
`kmeans_lloyd.py`, so that the fit time doesn't depend on how many iterations
a seed needs. The instrumented wrapper records `init_time` and `lloyd_time`
separately.
"""
from _kmeans import generate_kmeans_cases


def generate_cases() -> list[dict]:
    return generate_kmeans_cases({"tol": 0, "max_iter": 100})
