from __future__ import annotations

import numpy as np
from glum import GeneralizedLinearRegressor

__all__ = ["GeneralizedLinearClassifier"]


class GeneralizedLinearClassifier(GeneralizedLinearRegressor):
    """glum's binomial GLM as a binary classifier: glum only has a regressor,
    whose `predict` returns the probability of the positive class."""

    def fit(self, X, y, **kwargs):
        if self.family != "binomial":
            raise ValueError(f"family must be 'binomial', got {self.family!r}")
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"binary targets only, got {len(self.classes_)} classes")
        return super().fit(X, y.astype(np.float64), **kwargs)

    def predict_proba(self, X):
        proba = super().predict(X)
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return self.classes_[(super().predict(X) > 0.5).astype(int)]
