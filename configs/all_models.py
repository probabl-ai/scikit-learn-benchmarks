from _utils.common import disable_profiling_for_array_api_gpu_cases
from _utils.implementations import implementations_for_pixi_env

from _synthetic_trees import generate_cases as generate_tree_cases
from _synthetic_linear import generate_cases as generate_linear_cases
from _real_datasets import generate_cases as generate_real_cases

from sklbench.config.utils import (
    filter_array_api_supported_cases_if_needed,
    filter_gpu_cases_if_unavailable,
)


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env()

    cases = []
    for implem in implementations:
        cases += generate_tree_cases(implem, tier='normal')
        cases += generate_linear_cases(implem, tier='normal')
        cases += generate_real_cases(implem, max_tier='normal')

    for case in cases:
        case.setdefault('bench', {})
        case['bench'] |= {'n_runs': 5}
        case['bench'].setdefault('time_limit', 300)
    disable_profiling_for_array_api_gpu_cases(cases)

    cases = list(filter_gpu_cases_if_unavailable(cases))
    cases = list(filter_array_api_supported_cases_if_needed(cases))

    # TMP: KMeans-only rerun, revert right after dispatching.
    cases = [c for c in cases if c.algorithm.estimator == 'KMeans']

    return cases
