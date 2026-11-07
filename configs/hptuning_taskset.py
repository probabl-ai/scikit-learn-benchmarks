"""
`configs/hptuning.py` with the CPU affinity swept instead of the outer `n_jobs`.

Each point of the `n_jobs` sweep becomes an affinity on that many physical
cores, siblings included. The search runs on it with `n_jobs=-1` (one worker
per logical CPU).
"""

import configs.hptuning as hptuning
from _utils.scaling import cpu_affinity_for_physical_cores

from sklbench.config import HPTuningCase


def generate_cases(strong_scaling: bool = False) -> list[HPTuningCase]:
    cases = []
    for case in hptuning.generate_cases(strong_scaling):
        cpu_affinity = cpu_affinity_for_physical_cores(case.hptuning.n_jobs, with_siblings=True)
        cases.append(case.model_copy(update={
            "bench": case.bench.model_copy(update={"cpu_affinity": cpu_affinity}),
            "hptuning": case.hptuning.model_copy(update={"n_jobs": -1}),
        }))
    return cases
