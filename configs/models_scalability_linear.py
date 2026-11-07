"""
`models_scalability.py` limited to the linear models, to compare OpenBLAS builds
on their thread-count sweep.
"""
from models_scalability import generate_cases as generate_models_scalability_cases


LINEAR_ESTIMATORS = {"Ridge", "LogisticRegression"}


def generate_cases() -> list[dict]:
    return [
        case for case in generate_models_scalability_cases()
        if case["algorithm"]["estimator"] in LINEAR_ESTIMATORS
    ]
