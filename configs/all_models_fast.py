from _common import disable_profiling_for_array_api_gpu_cases
from _implementations import implementations_for_pixi_env

from synthetic_trees import generate_cases as generate_tree_cases
from synthetic_linear import generate_cases as generate_linear_cases
from real_datasets import generate_cases as generate_real_cases

from sklbench.config.utils import filter_array_api_supported_cases_if_needed


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env()

    cases = []
    for implem in implementations:
        cases += generate_tree_cases(implem, tier='fast')
        cases += generate_linear_cases(implem, tier='fast')
        cases += generate_real_cases(implem, max_tier='fast')

    for case in cases:
        case.setdefault('bench', {})
        case['bench'] |= {'n_runs': 5, 'py_spy_profiling': True}
        case['bench'].setdefault('time_limit', 30)
    disable_profiling_for_array_api_gpu_cases(cases)

    cases = list(filter_array_api_supported_cases_if_needed(cases))

    # removed clustering cases, as it's enough with what's in real_datasets.py's KMeans cases
    return cases
