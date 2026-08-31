from __future__ import annotations

import contextlib as _contextlib
from dataclasses import dataclass as _dataclass
import time as _time

from sklearn.ensemble import (
    HistGradientBoostingClassifier as _HistGradientBoostingClassifier,
    HistGradientBoostingRegressor as _HistGradientBoostingRegressor,
)
from sklearn.ensemble._hist_gradient_boosting import (
    gradient_boosting as _hgbt_module,
)

__all__ = [
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
]


@_dataclass
class _HGBTTimings:
    preprocess_x: float = 0.0
    binning_train: float = 0.0
    binning_validation: float = 0.0
    bin_fit: float = 0.0
    bin_transform: float = 0.0
    grower_init: float = 0.0
    grow: float = 0.0
    make_predictor: float = 0.0
    compute_hist: float = 0.0
    find_split: float = 0.0
    apply_split: float = 0.0
    tree_n_threads: int | None = None


@_contextlib.contextmanager
def _instrument_hgbt(estimator):
    timings = _HGBTTimings()
    original_preprocess_x = estimator._preprocess_X
    original_bin_data = estimator._bin_data
    original_tree_grower = _hgbt_module.TreeGrower

    def timed_preprocess_x(*args, **kwargs):
        t0 = _time.perf_counter()
        try:
            return original_preprocess_x(*args, **kwargs)
        finally:
            timings.preprocess_x += _time.perf_counter() - t0

    def timed_bin_data(*args, **kwargs):
        # `_bin_data`'s signature varies by sklearn version (1.9+ added
        # `sample_weight`); `is_training_data` is always passed as a keyword
        # though, so forward args through rather than hardcoding a shape.
        is_training_data = kwargs["is_training_data"]
        bin_mapper = estimator._bin_mapper
        original_fit = bin_mapper.fit
        original_transform = bin_mapper.transform

        def timed_fit(*args, **kwargs):
            t0 = _time.perf_counter()
            try:
                return original_fit(*args, **kwargs)
            finally:
                timings.bin_fit += _time.perf_counter() - t0

        def timed_transform(*args, **kwargs):
            t0 = _time.perf_counter()
            try:
                return original_transform(*args, **kwargs)
            finally:
                timings.bin_transform += _time.perf_counter() - t0

        bin_mapper.fit = timed_fit
        bin_mapper.transform = timed_transform
        t0 = _time.perf_counter()
        try:
            result = original_bin_data(*args, **kwargs)
            dt = _time.perf_counter() - t0
            if is_training_data:
                timings.binning_train += dt
            else:
                timings.binning_validation += dt
            return result
        finally:
            bin_mapper.fit = original_fit
            bin_mapper.transform = original_transform

    class TimedTreeGrower(original_tree_grower):
        def __init__(self, *args, **kwargs):
            t0 = _time.perf_counter()
            try:
                super().__init__(*args, **kwargs)
            finally:
                timings.grower_init += _time.perf_counter() - t0
                # Set once per fit() call and reused across all boosting
                # iterations/classes, so every TreeGrower sees the same value.
                timings.tree_n_threads = self.n_threads

        def grow(self):
            t0 = _time.perf_counter()
            try:
                return super().grow()
            finally:
                timings.grow += _time.perf_counter() - t0
                timings.compute_hist += self.total_compute_hist_time
                timings.find_split += self.total_find_split_time
                timings.apply_split += self.total_apply_split_time

        def make_predictor(self, *args, **kwargs):
            t0 = _time.perf_counter()
            try:
                return super().make_predictor(*args, **kwargs)
            finally:
                timings.make_predictor += _time.perf_counter() - t0

    estimator._preprocess_X = timed_preprocess_x
    estimator._bin_data = timed_bin_data
    _hgbt_module.TreeGrower = TimedTreeGrower
    try:
        yield timings
    finally:
        estimator._preprocess_X = original_preprocess_x
        estimator._bin_data = original_bin_data
        _hgbt_module.TreeGrower = original_tree_grower


class _InstrumentedHistGradientBoostingMixin:
    def fit(self, X, y, sample_weight=None):
        t0 = _time.perf_counter()
        with _instrument_hgbt(self) as timings:
            result = super().fit(X, y, sample_weight=sample_weight)
        total = _time.perf_counter() - t0
        binning = timings.binning_train + timings.binning_validation
        tree = timings.grower_init + timings.grow + timings.make_predictor

        self.total_time_ = total
        self.other_time_ = (
            total - binning - tree + timings.make_predictor - timings.preprocess_x
        )
        self.preprocess_time_ = timings.preprocess_x
        self.binning_time_ = binning
        self.binning_train_time_ = timings.binning_train
        self.binning_validation_time_ = timings.binning_validation
        self.bin_fit_time_ = timings.bin_fit
        self.bin_transform_time_ = timings.bin_transform
        self.tree_time_ = tree
        self.grower_init_time_ = timings.grower_init
        self.grow_time_ = timings.grow
        self.make_predictor_time_ = timings.make_predictor
        self.hist_time_ = timings.compute_hist
        self.find_split_time_ = timings.find_split
        self.apply_split_time_ = timings.apply_split
        self.tree_n_threads_ = timings.tree_n_threads
        return result


class HistGradientBoostingClassifier(
    _InstrumentedHistGradientBoostingMixin, _HistGradientBoostingClassifier
):
    pass


class HistGradientBoostingRegressor(
    _InstrumentedHistGradientBoostingMixin, _HistGradientBoostingRegressor
):
    pass
