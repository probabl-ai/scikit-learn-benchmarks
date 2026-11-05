"""
Strong scaling version of `configs/hptuning_taskset.py`.

The search always does `K * N_CORES` fits, and the CPU affinity sweeps a quarter,
half and all of the physical cores. In `hptuning_taskset.py`, the fits grow with
the cores instead (weak scaling).
"""

import configs.hptuning_taskset as hptuning_taskset

from sklbench.config import HPTuningCase

# The 3 runs of Ridge on susy with 22 cores on the 172-core Xeon take ~800 s.
TIME_LIMIT = 1200


def generate_cases() -> list[HPTuningCase]:
    return [
        case.model_copy(update={
            "bench": case.bench.model_copy(update={"time_limit": TIME_LIMIT}),
        })
        for case in hptuning_taskset.generate_cases(strong_scaling=True)
    ]
