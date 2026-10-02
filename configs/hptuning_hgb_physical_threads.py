"""
Temporary: the HistGradientBoosting searches of `configs/hptuning.py`, with
each loky worker limited to `physical_cores // n_jobs` threads instead of
joblib's default `logical_cpus // n_jobs` (2x more threads with SMT).

HGB on its own only uses physical cores, so this checks whether the joblib
default hurts it under nested parallelism. Only meaningful on SMT machines.
"""

from joblib import cpu_count

from hptuning import generate_cases as hptuning_cases


def generate_cases():
    n_cores = cpu_count(only_physical_cores=True)
    cases = []
    for case in hptuning_cases():
        if not case.algorithm.estimator.startswith("HistGradientBoosting"):
            continue
        n_threads = max(n_cores // case.hptuning.n_jobs, 1)
        hptuning = case.hptuning.model_copy(update={"inner_max_num_threads": n_threads})
        cases.append(case.model_copy(update={"hptuning": hptuning}))
    return cases
