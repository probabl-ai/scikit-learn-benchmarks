"""The 1-thread cases of `hgb_scalability.py`, pinned to a single logical CPU
(of a P-core on hybrid machines)."""

from pathlib import Path

from _utils.scaling import cpu_affinity_for_physical_cores, has_hybrid_cores
from hgb_scalability import generate_cases as generate_scalability_cases


def _one_cpu() -> list[int]:
    if has_hybrid_cores():
        p_cores = Path("/sys/devices/cpu_core/cpus").read_text().strip()
        return [int(p_cores.split(",")[0].split("-")[0])]
    return cpu_affinity_for_physical_cores(1)


def generate_cases() -> list[dict]:
    cpu_affinity = _one_cpu()
    return [
        {**case, "bench": {**case["bench"], "cpu_affinity": cpu_affinity}}
        for case in generate_scalability_cases()
        if case["bench"]["env"]["OMP_NUM_THREADS"] == "1"
    ]
