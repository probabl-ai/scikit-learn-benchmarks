import os
from pathlib import Path

import joblib


def _read_cpu_topology_id(cpu_id: int, name: str) -> str | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology/{name}")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _affinity_cpu_ids() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(joblib.cpu_count(only_physical_cores=False)))


def _logical_cpus_by_physical_core() -> list[list[int]]:
    cpu_ids = _affinity_cpu_ids()
    core_groups = {}
    for cpu_id in cpu_ids:
        package_id = _read_cpu_topology_id(cpu_id, "physical_package_id")
        core_id = _read_cpu_topology_id(cpu_id, "core_id")
        if package_id is None or core_id is None:
            return [[cpu_id] for cpu_id in cpu_ids]

        core_key = (package_id, core_id)
        core_groups.setdefault(core_key, []).append(cpu_id)
    return sorted(
        (sorted(cpu_group) for cpu_group in core_groups.values()),
        key=lambda cpu_group: cpu_group[0],
    )


def get_n_cores_list(*, max_n_cores: int | None = None) -> list[int]:
    n_cores = joblib.cpu_count(only_physical_cores=True)
    max_n_cores = min(max_n_cores or n_cores, n_cores)
    counts = []
    thread_count = 1
    while thread_count < max_n_cores:
        counts.append(thread_count)
        thread_count *= 2
    counts.append(max_n_cores)
    return counts


def has_hybrid_cores() -> bool:
    """Whether this machine's CPU mixes core types (e.g. Intel
    Performance/Efficiency cores), detected via the `cpu_core`/`cpu_atom`
    PMU sysfs directories the kernel exposes for Intel Hybrid Technology
    (absent on homogeneous CPUs, older kernels, and non-Intel machines,
    which this conservatively treats as non-hybrid).

    `OMP_PROC_BIND=true` has each OpenMP thread bind to one place chosen
    from across the whole affinity mask's topology, without distinguishing
    core type - on a hybrid CPU that can place a thread team smaller than
    the affinity mask on slower E-cores instead of packing it onto the
    faster P-cores the mask was sized for, even though a narrower mask
    (e.g. from a smaller thread count's own taskset) would have kept the
    same-size team entirely on P-cores - observed on a hybrid laptop as a
    ~2x fit-time regression between two thread counts that both resolved
    to the same actual tree_n_threads.
    """
    return (
        Path("/sys/devices/cpu_core/cpus").exists()
        and Path("/sys/devices/cpu_atom/cpus").exists()
    )


def taskset_for_physical_cores(n_cores: int, with_siblings: bool = False) -> str:
    # If n_cores == 1 and the first physical core has two logical CPUs,
    # this returns both logical CPU ids, for instance "0,1".
    if n_cores < 1:
        raise ValueError("n_cores must be at least 1")
    cpu_groups = _logical_cpus_by_physical_core()
    if with_siblings:
        # I think this should be the good way, but it doesn't work well
        # with joblib
        selected_cpus = [
            cpu_id
            for cpu_group in cpu_groups[:n_cores]
            for cpu_id in cpu_group
        ]
    else:
        selected_cpus = [
            cpu_group[0]
            for cpu_group in cpu_groups[:n_cores]
        ]
    return ",".join(str(cpu_id) for cpu_id in selected_cpus)

