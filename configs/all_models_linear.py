"""
`all_models.py` limited to the linear models, for the Pixi envs on a custom
OpenBLAS build. A strict subset (nothing added or resized), so every remaining
case is identical to its `all_models.py` counterpart and the cross comparison
pairs them with the other envs.
"""
from all_models import generate_cases as generate_all_models_cases
from _synthetic_linear import ALGORITHM_VARIANTS

from sklbench.config import EstimatorCase


LINEAR_ESTIMATORS = {variant["estimator"] for variant in ALGORITHM_VARIANTS}


def generate_cases() -> list[EstimatorCase]:
    return [
        case for case in generate_all_models_cases()
        if case.algorithm.estimator in LINEAR_ESTIMATORS
    ]
