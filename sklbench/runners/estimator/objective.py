"""Training objective of penalized linear models, in one convention for every
library, so that their fits of the same problem can be compared:

    mean loss + alpha * (l1_ratio * |w|_1 + (1 - l1_ratio) / 2 * |w|_2^2)

with the intercept not penalized. The loss is the log loss for logistic
regression, and half the unit deviance for the Poisson and Gamma GLMs. This is
scikit-learn's `PoissonRegressor` and glum's convention. `LogisticRegression`'s
`C` is `1 / (alpha * n_samples)`.
"""
import hashlib

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp


def _raw_prediction(estimator, X) -> tuple[np.ndarray, np.ndarray]:
    coef = np.asarray(estimator.coef_, dtype=np.float64)
    intercept = np.asarray(estimator.intercept_, dtype=np.float64)
    if coef.ndim == 2 and coef.shape[0] == 1:
        # Binary LogisticRegression: one row of coefficients.
        coef, intercept = coef[0], intercept.reshape(-1)[0]
    return np.asarray(X @ coef.T) + intercept, coef


def mean_loss(loss: str, y, raw: np.ndarray, classes=None) -> float:
    y = np.asarray(y)
    if loss == "binomial":
        y = np.searchsorted(classes, y) if classes is not None else y
        return float(np.mean(np.logaddexp(0, raw) - y * raw))
    if loss == "multinomial":
        y = np.searchsorted(classes, y)
        return float(np.mean(logsumexp(raw, axis=1) - raw[np.arange(len(y)), y]))
    y = y.astype(np.float64)
    if loss == "poisson":
        positive = y > 0
        y_log_y = np.zeros_like(y)
        y_log_y[positive] = y[positive] * (np.log(y[positive]) - raw[positive])
        return float(np.mean(y_log_y - y + np.exp(raw)))
    if loss == "gamma":
        return float(np.mean(raw - np.log(y) + y * np.exp(-raw) - 1))
    raise ValueError(f"Unknown loss {loss!r}")


def objective_metrics(
    estimator, X_train, y_train, X_test, y_test, *, loss: str, alpha: float, l1_ratio: float = 0
) -> tuple[dict, dict]:
    """`(fit metrics, predict metrics)`: the training objective and mean loss,
    and the test mean loss."""
    classes = getattr(estimator, "classes_", None)
    raw_train, coef = _raw_prediction(estimator, X_train)
    train_loss = mean_loss(loss, y_train, raw_train, classes)
    penalty = alpha * (
        l1_ratio * np.abs(coef).sum() + (1 - l1_ratio) / 2 * np.square(coef).sum()
    )
    raw_test, _ = _raw_prediction(estimator, X_test)
    return (
        {"objective": train_loss + float(penalty), "mean loss": train_loss},
        {"mean loss": mean_loss(loss, y_test, raw_test, classes)},
    )


def data_hash(X, y) -> str:
    """Hash of the training data, to compare objectives only between fits of
    the exact same matrix (e.g. not across scikit-learn versions whose
    preprocessing differs)."""
    digest = hashlib.blake2b(digest_size=8)
    if sp.issparse(X):
        X = X.tocsr()
        for array in (X.data, X.indices, X.indptr):
            digest.update(np.ascontiguousarray(array).tobytes())
    else:
        digest.update(np.ascontiguousarray(X).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()
