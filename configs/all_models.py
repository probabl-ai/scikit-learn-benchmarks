from _implementations import implementations_for_pixi_env

from synthetic_trees import generate_cases as generate_tree_cases
from synthetic_linear import generate_cases as generate_linear_cases
from real_datasets import generate_cases as generate_real_cases

from sklbench.config.utils import filter_array_api_supported_cases_if_needed


def generate_cases() -> list[dict]:
    implementations = implementations_for_pixi_env()

    cases = []
    for implem in implementations:
        cases += generate_tree_cases(implem, tier='normal')
        cases += generate_linear_cases(implem, tier='normal')
        cases += generate_real_cases(implem, max_tier='normal')

    cases = list(filter_array_api_supported_cases_if_needed(cases))

    return cases
