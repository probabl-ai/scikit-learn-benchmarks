from __future__ import annotations

from joblib import cpu_count as _cpu_count
from xgboost import (
    XGBClassifier as _XGBClassifier,
    XGBRegressor as _XGBRegressor,
)

__all__ = [
    "XGBClassifier",
    "XGBRegressor",
]


class _PhysicalCoresMixin:
    # XGBoost defaults to one thread per logical CPU. Default to the physical
    # core count instead, like scikit-learn's OpenMP code and LightGBM, so a
    # comparison doesn't hinge on SMT.
    def fit(self, X, y, **kwargs):
        if self.n_jobs is None:
            self.n_jobs = _cpu_count(only_physical_cores=True)
        self.n_threads_ = self.n_jobs
        return super().fit(X, y, **kwargs)


class XGBClassifier(_PhysicalCoresMixin, _XGBClassifier):
    pass


class XGBRegressor(_PhysicalCoresMixin, _XGBRegressor):
    pass
