"""
Experiment: `configs/hptuning_all_models.py` with fewer workers than logical CPUs.

The searches run under 3 CPU affinities (a quarter, half and all of the
physical cores, siblings included) and, for each, with `n_jobs` set to the
number of logical CPUs of the affinity divided by 1, 2, 4 and 8. On the
172-core Xeon: 43 cores, one socket and the whole machine.

With fewer workers than logical CPUs, joblib gives each worker
`n_cpus // n_jobs` OpenMP and BLAS threads.

`HPTUNING_N_JOBS_POINT=<n_cores>:<divisor>` keeps a single point of the sweep,
to run and publish the points one at a time.
"""

import os

import configs.hptuning_all_models as hptuning_all_models
from _utils.scaling import cpu_affinity_for_physical_cores
from joblib import cpu_count

from sklbench.config import HPTuningCase

N_CORES = cpu_count(only_physical_cores=True)
# Sets, since small machines give the same value several times.
N_CORES_LIST = sorted({max(N_CORES // 4, 1), max(N_CORES // 2, 1), N_CORES}, reverse=True)
DIVISORS = [1, 2, 4, 8]


def sweep_points() -> list[tuple[int, int]]:
    """(n_cores, divisor) points, without those giving an `n_jobs` already listed."""
    points = []
    for n_cores in N_CORES_LIST:
        n_cpus = len(cpu_affinity_for_physical_cores(n_cores, with_siblings=True))
        seen = set()
        for divisor in DIVISORS:
            n_jobs = max(n_cpus // divisor, 1)
            if n_jobs not in seen:
                seen.add(n_jobs)
                points.append((n_cores, divisor))
    return points


def generate_cases() -> list[HPTuningCase]:
    points = sweep_points()
    only = os.environ.get("HPTUNING_N_JOBS_POINT")
    if only:
        point = tuple(int(value) for value in only.split(":"))
        if point not in points:
            raise ValueError(f"HPTUNING_N_JOBS_POINT={only} is not one of {points}")
        points = [point]

    cases = []
    for n_cores, divisor in points:
        cpu_affinity = cpu_affinity_for_physical_cores(n_cores, with_siblings=True)
        n_jobs = max(len(cpu_affinity) // divisor, 1)
        for case in hptuning_all_models.generate_cases():
            cases.append(case.model_copy(update={
                "bench": case.bench.model_copy(update={"cpu_affinity": cpu_affinity}),
                "hptuning": case.hptuning.model_copy(update={"n_jobs": n_jobs}),
            }))
    return cases
