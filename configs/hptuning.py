"""
Hyperparameter-tuning scalability cases: `RandomizedSearchCV` over a few
real datasets already used by `configs/real_datasets.py`, tuning one linear
model (LogisticRegression for the classification datasets, RidgeCV for the
regression one) and HistGradientBoosting per dataset.

Datasets with many rows/high-cardinality categoricals are subsampled (via
`hptuning.max_samples`) so each case's 14-point search finishes in well
under a minute on a 14-physical-core box, at the default
`outer_n_jobs = round(sqrt(physical_cores))` (see
`sklbench.config.models.hptuning.resolve_outer_n_jobs`). `outer_n_jobs`/
`inner_n_jobs` can be set explicitly per case to override that balance.
"""

import numpy as np

from sklbench.config import Algorithm, Data, HPTuning, HPTuningCase

BENCH = {"n_runs": 3}

# Keys are full pipeline param paths - "estimator__..." here since these
# tune the model itself, not preprocessing (see the runner's
# `_build_pipeline` for the "preprocessor"/"estimator" step names).
_HGB_PARAM_DISTRIBUTIONS = {
    "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    "estimator__max_leaf_nodes": [7, 15, 31, 63],
    "estimator__min_samples_leaf": [5, 10, 20, 50],
    "estimator__l2_regularization": [0.0, 0.1, 0.3, 1.0],
    "estimator__max_iter": [50, 100, 150],
}
_HGB_PARAMS = {"early_stopping": False}

# Not "balanced": see `sklbench.config.utils` callers / repo notes - it
# distorts calibration without moving ROC AUC.
_LOGISTIC_REGRESSION_PARAMS = {"solver": "lbfgs", "max_iter": 1000}
_LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS = {
    "estimator__C": list(np.logspace(-3, 3, 13)),
    "estimator__fit_intercept": [True, False],
}


def _case(
    dataset: str,
    estimator: str,
    estimator_params: dict,
    param_distributions: dict,
    max_samples: int | None = None,
) -> HPTuningCase:
    return HPTuningCase(
        bench=BENCH,
        algorithm=Algorithm(estimator=estimator, estimator_params=estimator_params),
        data=Data(dataset=dataset),
        hptuning=HPTuning(
            param_distributions=param_distributions,
            max_samples=max_samples,
        ),
    )


def generate_cases() -> list[HPTuningCase]:
    cases = []

    # ames_housing: regression, 1460 rows, mixed numeric/categorical with
    # real missing values - small enough to use in full. RidgeCV already
    # picks its own alpha internally (efficient built-in LOO-CV over a
    # fixed, wide `alphas` grid, set below), so the outer RandomizedSearchCV
    # instead tunes *preprocessing* hyperparameters - how the
    # OneHotEncoder collapses rare/high-cardinality categories, and how
    # numeric missing values are imputed.
    cases.append(
        _case(
            "ames_housing",
            "RidgeCV",
            {"alphas": list(np.logspace(-4, 4, 19))},
            {
                "preprocessor__categorical__max_categories": [5, 10, 20, 50],
                "preprocessor__categorical__min_frequency": [1, 2, 5, 10, 20],
                "preprocessor__numeric__simpleimputer__strategy": ["mean", "median"],
            },
        )
    )
    cases.append(
        _case(
            "ames_housing",
            "HistGradientBoostingRegressor",
            _HGB_PARAMS,
            _HGB_PARAM_DISTRIBUTIONS,
        )
    )

    # amazon_employee_access: classification, 32769 rows, 9 all-categorical
    # features (up to 7518 uniques) - subsampled to bound OneHotEncoder's
    # output width and per-candidate fit time.
    for estimator, estimator_params, param_distributions in [
        (
            "LogisticRegression",
            _LOGISTIC_REGRESSION_PARAMS,
            _LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
        ),
        ("HistGradientBoostingClassifier", _HGB_PARAMS, _HGB_PARAM_DISTRIBUTIONS),
    ]:
        cases.append(
            _case(
                "amazon_employee_access",
                estimator,
                estimator_params,
                param_distributions,
                max_samples=8000,
            )
        )

    # kddcup09_churn: classification, 50000 rows, 207 columns (mostly
    # numeric with heavy missingness, some high-cardinality categorical),
    # ~7.3% positive rate - subsampled (stratified in `_subsample`) to
    # bound both row count and the cost of imputing/encoding 207 columns
    # per candidate fit.
    for estimator, estimator_params, param_distributions in [
        (
            "LogisticRegression",
            _LOGISTIC_REGRESSION_PARAMS,
            _LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
        ),
        ("HistGradientBoostingClassifier", _HGB_PARAMS, _HGB_PARAM_DISTRIBUTIONS),
    ]:
        cases.append(
            _case(
                "kddcup09_churn",
                estimator,
                estimator_params,
                param_distributions,
                max_samples=6000,
            )
        )

    return cases
