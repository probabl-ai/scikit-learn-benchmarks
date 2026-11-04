"""
`configs/hptuning.py` with the CPU affinity swept instead of the outer `n_jobs`.

Each point of the `n_jobs` sweep becomes an affinity on that many physical
cores, siblings included. The search runs on it with `n_jobs=-1` (one worker
per logical CPU) and, if the cores have SMT siblings, also with one worker per
physical core.
"""

import configs.hptuning as hptuning
from _utils.scaling import cpu_affinity_for_physical_cores

from sklbench.config import HPTuningCase


def generate_cases() -> list[HPTuningCase]:
    cases = []
    for case in hptuning.generate_cases():
        n_physical_cores = case.hptuning.n_jobs
        cpu_affinity = cpu_affinity_for_physical_cores(n_physical_cores, with_siblings=True)
        bench = case.bench.model_copy(update={"cpu_affinity": cpu_affinity})
        # Not `cpu_count(only_physical_cores=True)` in the runner: under an
        # affinity it returns the logical CPU count, the same as `n_jobs=-1`
        # (https://github.com/joblib/loky/pull/651).
        n_jobs_list = [-1, n_physical_cores] if len(cpu_affinity) > n_physical_cores else [-1]
        for n_jobs in n_jobs_list:
            cases.append(case.model_copy(update={
                "bench": bench,
                "hptuning": case.hptuning.model_copy(update={"n_jobs": n_jobs}),
            }))
    return cases
