from __future__ import annotations

import contextlib as _contextlib
import time as _time

from sklearn.ensemble import (
    RandomForestClassifier as _RandomForestClassifier,
    RandomForestRegressor as _RandomForestRegressor,
)
from sklearn.ensemble import _forest as _forest_module

__all__ = [
    "RandomForestClassifier",
    "RandomForestRegressor",
]


@_contextlib.contextmanager
def _instrument_forest():
    # One (start, end) pair per tree, appended by whichever worker thread
    # finished it. No lock: CPython's GIL makes `list.append` atomic, and
    # locking here - once per tree, from every worker - would itself become
    # a contention point at high thread counts, skewing the very
    # thread-scaling behavior this wrapper is meant to measure.
    tree_intervals: list[tuple[float, float]] = []
    original_parallel_build_trees = _forest_module._parallel_build_trees

    def timed_parallel_build_trees(*args, **kwargs):
        t0 = _time.perf_counter()
        try:
            return original_parallel_build_trees(*args, **kwargs)
        finally:
            tree_intervals.append((t0, _time.perf_counter()))

    _forest_module._parallel_build_trees = timed_parallel_build_trees
    try:
        yield tree_intervals
    finally:
        _forest_module._parallel_build_trees = original_parallel_build_trees


class _InstrumentedForestMixin:
    def fit(self, X, y, sample_weight=None):
        t0 = _time.perf_counter()
        with _instrument_forest() as tree_intervals:
            result = super().fit(X, y, sample_weight=sample_weight)
        total = _time.perf_counter() - t0

        # `build_trees_time_` is the wall-clock span from the first tree's
        # start to the last tree's finish - i.e. the actual duration of the
        # `Parallel(...)` block, however many workers ran concurrently.
        # `build_trees_cpu_time_` sums each tree's own duration regardless of
        # overlap, so `build_trees_cpu_time_ / build_trees_time_` is a rough
        # parallel-efficiency ratio (close to `n_jobs` when trees scale well).
        n_trees_built = len(tree_intervals)
        build_trees_cpu_time = sum(end - start for start, end in tree_intervals)
        build_trees_time = (
            max(end for _, end in tree_intervals)
            - min(start for start, _ in tree_intervals)
            if tree_intervals
            else 0.0
        )

        self.total_time_ = total
        self.build_trees_time_ = build_trees_time
        self.build_trees_cpu_time_ = build_trees_cpu_time
        self.n_trees_built_ = n_trees_built
        self.avg_tree_time_ = (
            build_trees_cpu_time / n_trees_built if n_trees_built else 0.0
        )
        self.other_time_ = total - build_trees_time
        return result


class RandomForestClassifier(_InstrumentedForestMixin, _RandomForestClassifier):
    pass


class RandomForestRegressor(_InstrumentedForestMixin, _RandomForestRegressor):
    pass
