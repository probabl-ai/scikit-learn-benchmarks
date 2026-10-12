"""Subset of `all_models.py` limited to certain estimators with cuML wrapper. 
(They were defined by orgisel in the labs-nvidia slack channel.)

A strict subset (nothing added or resized), so every remaining case is
identical to its `all_models.py` counterpart. Run it for both environments
when the scikit-learn cases should match the cuML cases:

    ./run.sh cuml sklearn-pypi --config configs/all_models_cuml_only.py

`all_models.py` still drops unsupported cases for `library: "cuml"` only.
Other libraries keep the full matrix there.
"""

from all_models import generate_cases as generate_all_models_cases

from sklbench.config import EstimatorCase
from sklbench.config.utils import is_cuml_supported_case


def generate_cases() -> list[EstimatorCase]:
    return [
        case for case in generate_all_models_cases() if is_cuml_supported_case(case)
    ]
