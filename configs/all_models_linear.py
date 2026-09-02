from _common import disable_profiling_for_array_api_gpu_cases
from _implementations import implementations_for_pixi_env

from synthetic_linear import generate_cases as generate_linear_cases

from sklbench.config.utils import (
    filter_array_api_supported_cases_if_needed,
    filter_gpu_cases_if_unavailable,
)


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env()

    cases = []
    for implem in implementations:
        cases += generate_linear_cases(implem, tier='normal')

    for case in cases:
        case.setdefault('bench', {})
        case['bench'] |= {'n_runs': 5}
        case['bench'].setdefault('time_limit', 300)
    disable_profiling_for_array_api_gpu_cases(cases)

    cases = list(filter_gpu_cases_if_unavailable(cases))
    cases = list(filter_array_api_supported_cases_if_needed(cases))

    return cases
