from __future__ import annotations

from catboost import (
    CatBoostClassifier as _CatBoostClassifier,
    CatBoostRegressor as _CatBoostRegressor,
)
from joblib import cpu_count as _cpu_count
import pandas as _pd

__all__ = [
    "CatBoostClassifier",
    "CatBoostRegressor",
]

# The "catboost" preprocessing leaves unknown categories as NaN, which CatBoost
# rejects in categorical columns.
_UNKNOWN_CATEGORY = -2


def _to_catboost_input(X):
    """CatBoost ignores pandas' `category` dtype: categorical columns must be
    passed as `cat_features` and hold ints or strings. Assumes the categories
    are ordinal codes, as produced by the "catboost" preprocessing."""
    if not isinstance(X, _pd.DataFrame):
        return X, []
    cat_features = list(X.select_dtypes(["category"]).columns)
    if not cat_features:
        return X, []
    X = X.copy()
    for column in cat_features:
        X[column] = (
            X[column].astype("float64").fillna(_UNKNOWN_CATEGORY).astype("int64")
        )
    return X, cat_features


class _CatBoostAdapterMixin:
    def fit(self, X, y, **kwargs):
        # CatBoost defaults to one thread per logical CPU, see
        # `gbdt_xgboost._PhysicalCoresMixin`.
        if self.get_param("thread_count") is None:
            self.set_params(thread_count=_cpu_count(only_physical_cores=True))
        self.n_threads_ = self.get_param("thread_count")
        X, cat_features = _to_catboost_input(X)
        return super().fit(X, y, cat_features=cat_features or None, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(_to_catboost_input(X)[0], **kwargs)


class CatBoostClassifier(_CatBoostAdapterMixin, _CatBoostClassifier):
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(_to_catboost_input(X)[0], **kwargs)


class CatBoostRegressor(_CatBoostAdapterMixin, _CatBoostRegressor):
    pass
