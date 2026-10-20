"""KMeans matrix of `_kmeans.py` that times the k-means++ init.

Each fit runs a single Lloyd iteration (`max_iter` can't be 0), so its time is
mostly the k-means++ init, plus input validation and that one iteration. The
instrumented wrapper also records `init_time` alone.
"""
from _kmeans import generate_kmeans_cases


def generate_cases() -> list[dict]:
    return generate_kmeans_cases({"max_iter": 1})
