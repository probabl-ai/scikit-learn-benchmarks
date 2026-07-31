from _implementations import implementations_for_pixi_env

from synthetic_trees import generate_cases as generate_tree_cases
from synthetic_linear import generate_cases as generate_linear_cases
from real_datasets import generate_cases as generate_real_cases

from sklbench.config.utils import filter_array_api_supported_cases_if_needed


def get_basic_kmeans_case(implem):
    return {
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": {"n_init": 1},
        },
        "data": {
            "source": "make_blobs",
            "generation_kwargs": {"centers": 5, "n_samples": 1000},
        },
        "implementation": implem
    }


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env()

    cases = []
    for implem in implementations:
        cases.append(get_basic_kmeans_case(implem))
        cases += generate_tree_cases(implem, tier='test')
        cases += generate_linear_cases(implem, tier='test')
        cases += generate_real_cases(implem, max_tier='test')

    for case in cases:
        case.setdefault('bench', {})
        case['bench'] |= {'n_runs': 1, 'py_spy_profiling': True}
        case['bench'].setdefault('time_limit', 2)

    cases = list(filter_array_api_supported_cases_if_needed(cases))

    return cases
