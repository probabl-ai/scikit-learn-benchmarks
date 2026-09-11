"""NUMA topology helpers for pinning a case's benchmark subprocess to one
node via the existing `bench.cpu_affinity` mechanism.

Pinning to one node's cores is enough on its own to also confine memory
allocation to that node: Linux's default (first-touch) memory policy
allocates each page on whichever CPU faults it in, and a CPU-affinity mask
restricted to that node's cores guarantees that CPU is one of them - no
separate `numactl --membind` needed. Confirmed empirically on an 8-node
runner: pinning both compute and memory via `numactl --cpunodebind=N
--membind=N` and pinning compute alone via `taskset -c <node N's cores>`
produced statistically indistinguishable timings, both far tighter than
leaving placement to the OS - see
https://github.com/probabl-ai/scikit-learn-benchmarks/issues/80.
"""
from pathlib import Path

NUMA_NODES_SYSFS = Path("/sys/devices/system/node")


def numa_node_count() -> int:
    if not NUMA_NODES_SYSFS.is_dir():
        return 1
    return sum(1 for p in NUMA_NODES_SYSFS.glob("node[0-9]*") if p.is_dir()) or 1


def _parse_cpu_list(cpu_list: str) -> list[int]:
    """Parse a Linux sysfs `cpulist` string ("0-21,172-193") into CPU ids."""
    cpu_ids: list[int] = []
    for part in cpu_list.split(","):
        if "-" in part:
            start, end = part.split("-")
            cpu_ids.extend(range(int(start), int(end) + 1))
        else:
            cpu_ids.append(int(part))
    return cpu_ids


def cpu_affinity_for_numa_node(node: int) -> list[int]:
    """CPU ids for NUMA node `node`, read from sysfs."""
    cpu_list = (NUMA_NODES_SYSFS / f"node{node}" / "cpulist").read_text(encoding="utf-8").strip()
    return _parse_cpu_list(cpu_list)


def auto_numa_cpu_affinity(node: int = 0) -> list[int] | None:
    """CPU ids pinning to one NUMA node, or None on a single-node host
    where doing so would only discard cores for no benefit.
    """
    if numa_node_count() <= 1:
        return None
    return cpu_affinity_for_numa_node(node)
