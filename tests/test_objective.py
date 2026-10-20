import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import GammaRegressor, LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, mean_gamma_deviance, mean_poisson_deviance

from sklbench.runners.estimator.objective import data_hash, objective_metrics


@pytest.mark.parametrize("n_classes", [2, 3])
def test_logistic_regression_objective(n_classes):
    X, y = make_classification(n_samples=300, n_informative=4, n_classes=n_classes, random_state=0)
    alpha = 1e-2
    estimator = LogisticRegression(C=1 / (alpha * len(y)), tol=1e-10).fit(X, y)
    loss = "binomial" if n_classes == 2 else "multinomial"
    fit, predict = objective_metrics(estimator, X, y, X, y, loss=loss, alpha=alpha)
    assert fit["mean loss"] == pytest.approx(log_loss(y, estimator.predict_proba(X)))
    penalty = alpha / 2 * np.square(estimator.coef_).sum()
    assert fit["objective"] == pytest.approx(fit["mean loss"] + penalty)
    assert predict["mean loss"] == pytest.approx(fit["mean loss"])


@pytest.mark.parametrize(
    "estimator_class, loss, deviance",
    [(PoissonRegressor, "poisson", mean_poisson_deviance), (GammaRegressor, "gamma", mean_gamma_deviance)],
)
def test_glm_objective_is_half_mean_deviance(estimator_class, loss, deviance):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.poisson(np.exp(X @ [0.3, -0.2, 0.1, 0.0]))
    if loss == "gamma":
        y = y + 1.0
    estimator = estimator_class(alpha=1e-2).fit(X, y)
    fit, _ = objective_metrics(estimator, X, y, X, y, loss=loss, alpha=1e-2)
    assert fit["mean loss"] == pytest.approx(deviance(y, estimator.predict(X)) / 2)


def test_data_hash_depends_on_values_not_layout():
    X = np.arange(12.0).reshape(4, 3)
    y = np.arange(4)
    assert data_hash(X, y) == data_hash(np.asfortranarray(X), y)
    X_changed = X.copy()
    X_changed[0, 0] += 1e-12
    assert data_hash(X, y) != data_hash(X_changed, y)
