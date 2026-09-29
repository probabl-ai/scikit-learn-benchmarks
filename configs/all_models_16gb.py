"""
`all_models.py` minus the cases that OOM on a 16GB machine. A strict subset
(nothing added or resized), so every remaining case is identical to its
`all_models.py` counterpart.

Cut-offs are based on peak system memory measured on a 16-thread / 31GB
machine (hardware `3b5e61`) running `all_models.py`. Peaks there run ~5-6x
the size of the dense float64 X, because generating and validating the data
makes copies:

- Synthetic linear: drop cases whose dense float64 X is over
  `MAX_ARRAY_BYTES` (e.g. Ridge 20M x 20 peaked at 21GB, LogisticRegression
  8M x 20 at 9GB). For Ridge/LinearRegression, also drop cases whose
  (n_features, n_features) Gram matrix is over that limit (LinearRegression
  2200 x 17900 peaked at 10GB and failed; Ridge 5000 x 40000 peaked at 12GB).
- Real datasets: drop Ridge on `year_prediction_msd`. Its spline expansion
  peaked at 7-15GB depending on the implementation.

Synthetic tree cases stay. Their memory scales with `n_estimators = 2 *
cpu_count()`, and they peaked at 5GB or less with 16 threads.
"""
from all_models import generate_cases as generate_all_models_cases

from sklbench.config import EstimatorCase


MAX_ARRAY_BYTES = 2**30

GRAM_ESTIMATORS = {"Ridge", "LinearRegression"}

EXCLUDED_REAL_CASES = {("year_prediction_msd", "Ridge")}


def _fits_in_16gb(case: EstimatorCase) -> bool:
    estimator = case.algorithm.estimator
    data = case.data
    if data.dataset is not None:
        return (data.dataset, estimator) not in EXCLUDED_REAL_CASES

    if data.source not in ("make_regression", "make_classification"):
        return True
    n_samples = data.generation_kwargs["n_samples"]
    n_features = data.generation_kwargs["n_features"]
    if n_samples * n_features * 8 > MAX_ARRAY_BYTES:
        return False
    if estimator in GRAM_ESTIMATORS and n_features**2 * 8 > MAX_ARRAY_BYTES:
        return False
    return True


def generate_cases() -> list[EstimatorCase]:
    return [case for case in generate_all_models_cases() if _fits_in_16gb(case)]
