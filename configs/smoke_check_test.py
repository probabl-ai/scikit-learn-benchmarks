import sys

from _common import disable_profiling_for_array_api_gpu_cases
from _implementations import implementations_for_pixi_env

from synthetic_trees import generate_cases as generate_tree_cases
from synthetic_linear import generate_cases as generate_linear_cases
from real_datasets import generate_cases as generate_real_cases
from hgb_scalability import generate_xs_cases as generate_hgb_scaling_cases

from sklbench.config.utils import (
    filter_array_api_supported_cases_if_needed,
    filter_gpu_cases_if_unavailable,
)


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

    # Not looped over `implementations` above: hgb_scalability.py's cases are
    # always plain sklearn (its thread-scaling/cpu_affinity sweep isn't an
    # array-API concept), so one copy exercises that path regardless of
    # which Pixi environment this config is running under.
    cases += generate_hgb_scaling_cases()

    benchs = [
        {'n_runs': 1, 'py_spy_profiling': True},
        {'n_runs': 1, 'py_spy_profiling': True, 'py_spy_native': False},
        {'n_runs': 1, 'py_spy_profiling': False, 'cprofile_profiling': True},
    ]
    if sys.platform != "darwin":
        # Exercises bench.cpu_affinity end to end (pinned via psutil in
        # sklbench/orchestrator/commands.py). Not supported on macOS -
        # pin_process_affinity raises there rather than silently no-op'ing
        # - so this case is deliberately not run there.
        benchs.append({'n_runs': 1, 'py_spy_profiling': False, 'cpu_affinity': [0]})
    benchs += [{'n_runs': 1, 'py_spy_profiling': False}] * len(cases)
    for case, bench in zip(cases, benchs):
        case.setdefault('bench', {})
        case['bench'] |= bench
        # 4s was too tight even for the first of these trivial synthetic
        # cases on a GitHub-hosted macOS runner: one-time interpreter/import
        # cold-start cost there (measured 3-11s across runs) can exceed it on
        # its own, before any actual fitting. 10s gives headroom without
        # making this "small exploratory matrix" meaningfully slower.
        case['bench'].setdefault('time_limit', 10)
    disable_profiling_for_array_api_gpu_cases(cases)

    cases = list(filter_gpu_cases_if_unavailable(cases))
    cases = list(filter_array_api_supported_cases_if_needed(cases))

    return cases
