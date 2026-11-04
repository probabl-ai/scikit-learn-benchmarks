"""Quality metrics must tell a perfect model from a useless one. They decide whether
two implementations are flagged as not equivalent on the dashboards."""

import numpy as np
import pytest

from sklbench.runners.estimator.metrics import get_subset_metrics_of_estimator


class Classifier:
    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, x):
        return self.proba


class Regressor:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, x):
        return self.predictions


class Clusterer:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, x):
        return self.labels


X = np.random.default_rng(0).normal(size=(200, 3))


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classification_metrics(n_classes):
    y = np.arange(200) % n_classes
    almost_sure = np.eye(n_classes)[y] * 0.97 + 0.03 / n_classes
    uniform = np.full((200, n_classes), 1 / n_classes)

    perfect = get_subset_metrics_of_estimator("classification", "inference", Classifier(almost_sure), (X, y))
    useless = get_subset_metrics_of_estimator("classification", "inference", Classifier(uniform), (X, y))

    assert perfect["ROC AUC"] == pytest.approx(1)
    assert useless["ROC AUC"] == pytest.approx(0.5)
    assert perfect["logloss"] < useless["logloss"]
    assert perfect["Brier score"] < useless["Brier score"]


def test_regression_metrics():
    y = np.linspace(0, 1, 200)

    perfect = get_subset_metrics_of_estimator("regression", "inference", Regressor(y), (X, y))
    mean = get_subset_metrics_of_estimator("regression", "inference", Regressor(np.full(200, y.mean())), (X, y))

    assert perfect["R2"] == pytest.approx(1) and perfect["RMSE"] == pytest.approx(0)
    assert mean["R2"] == pytest.approx(0) and mean["RMSE"] > 0


def test_clustering_metrics():
    y = np.arange(200) % 2

    perfect = get_subset_metrics_of_estimator("clustering", "inference", Clusterer(1 - y), (X, y))
    independent = np.arange(200) % 4 // 2
    useless = get_subset_metrics_of_estimator("clustering", "inference", Clusterer(independent), (X, y))

    assert perfect["homogeneity"] == pytest.approx(1)
    assert useless["homogeneity"] < 0.01
