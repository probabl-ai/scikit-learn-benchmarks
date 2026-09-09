"""NUMA topology helpers for pinning a case's benchmark subprocess to one
node via the existing `bench.taskset` mechanism.

Plain `taskset -c <node's core list>` is enough on its own to also confine
memory allocation to that node: Linux's default (first-touch) memory policy
allocates each page on whichever CPU faults it in, and taskset guarantees
that CPU is one of the node's own cores - no separate `numactl --membind`
needed. Confirmed empirically on an 8-node runner: pinning both compute and
memory via `numactl --cpunodebind=N --membind=N` and pinning compute alone
via `taskset -c <node N's cores>` produced statistically indistinguishable
timings, both far tighter than leaving placement to the OS - see
https://github.com/probabl-ai/scikit-learn-benchmarks/issues/80.
"""
from pathlib import Path

NUMA_NODES_SYSFS = Path("/sys/devices/system/node")


def numa_node_count() -> int:
    if not NUMA_NODES_SYSFS.is_dir():
        return 1
    return sum(1 for p in NUMA_NODES_SYSFS.glob("node[0-9]*") if p.is_dir()) or 1


def taskset_for_numa_node(node: int) -> str:
    """`taskset -c` core list for NUMA node `node`, read from sysfs."""
    return (NUMA_NODES_SYSFS / f"node{node}" / "cpulist").read_text(encoding="utf-8").strip()


def auto_numa_taskset(node: int = 0) -> str | None:
    """`taskset -c` core list pinning to one NUMA node, or None on a
    single-node host where doing so would only discard cores for no
    benefit.
    """
    if numa_node_count() <= 1:
        return None
    return taskset_for_numa_node(node)
