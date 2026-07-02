import os
from math import sqrt
from pathlib import Path

import joblib


DFT_MAX_ITER = 100
DFT_MAX_LEAF_NODES = 31
DFT_MAX_FEATURES = 0.5

TASKS = ["classification", "regression"]

COLUMNS_KINDS = ["continuous", "mix"]


WORKLOADS = [
    {
        "name": "XS",
        "n_samples": 1000,
        "n_features": 10,
        "max_features": 1.0,
    },
    {
        "name": "S-thin",
        "n_samples": 5_000,
        "n_features": 10,
        "max_features": 1.0,
    },
    {
        "name": "S",
        "n_samples": 1_000,
        "n_features": 100,
    },
    {
        "name": "M",
        "n_samples": 20_000,
        "n_features": 50,
    },
    {
        "name": "M-very-thin",
        "n_samples": 50_000,
        "n_features": 5,
        "max_features": 1.0,
    },
    {
        "name": "L-thin",
        "n_samples": 500_000,
        "n_features": 10,
        "max_features": 1.0,
    },
    {
        "name": "L",
        "n_samples": 100_000,
        "n_features": 100,
    },
    {
        "name": "M-stumps",
        "n_samples": 200_000,
        "n_features": 100,
        "max_leaf_nodes": 2,
    },
    {
        "name": "L-stumps",
        "n_samples": 1_000_000,
        "n_features": 100,
        "max_leaf_nodes": 2,
    },
    {
        "name": "L-leaf255",
        "n_samples": 100_000,
        "n_features": 100,
        "max_iter": 20,
        "max_leaf_nodes": 255,
    },
]


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


def _thread_counts() -> list[int]:
    physical_cpus = joblib.cpu_count(only_physical_cores=True)
    counts = []
    thread_count = 1
    while thread_count < physical_cpus:
        counts.append(thread_count)
        thread_count *= 2
    counts.append(physical_cpus)
    return counts


def taskset_for_physical_cores(n_cores: int) -> str:
    # If n_cores == 1 and the first physical core has two logical CPUs,
    # this returns both logical CPU ids, for instance "0,1".
    if n_cores < 1:
        raise ValueError("n_cores must be at least 1")
    cpu_groups = _logical_cpus_by_physical_core()
    if False:
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


def _case(workload: dict, task: str, columns_kind: str, thread_count: int) -> dict:
    is_classifier = task == "classification"
    estimator = (
        "HistGradientBoostingClassifier"
        if is_classifier
        else "HistGradientBoostingRegressor"
    )

    source = f"make_trees_{task}_data"
    default_n_informative = round(sqrt(workload["n_features"]))
    generation_kwargs = {
        "n_samples": workload["n_samples"],
        "n_features": workload["n_features"],
        "n_informative": workload.get("n_informative", default_n_informative),
        "random_state": 0,
        "n_redundant": 0,
        "columns": columns_kind,
    }
    if is_classifier:
        generation_kwargs.update({"n_classes": 2})

    return {
        "bench": {
            "n_runs": 2,
            "taskset": taskset_for_physical_cores(thread_count),
        },
        "implementation": {
            "library": "sklearn",
        },
        "metadata": {
            "task": task,
            "name": workload["name"],
            "columns_kind": columns_kind,
        },
        "algorithm": {
            "estimator": estimator,
            "estimator_params": {
                "max_iter": workload.get("max_iter", DFT_MAX_ITER),
                "max_leaf_nodes": workload.get("max_leaf_nodes", DFT_MAX_LEAF_NODES),
                "max_features": workload.get("max_features", DFT_MAX_FEATURES),
                "max_bins": 255,
                "early_stopping": False,
            },
        },
        "data": {
            "id": workload["name"],
            "source": source,
            "generation_kwargs": generation_kwargs,
            "split_kwargs": {"test_size": 0.2},
            "order": "C",
        },
    }


def generate_cases() -> list[dict]:
    return [
        _case(workload, task, columns_kind, thread_count)
        for workload in WORKLOADS
        for task in TASKS
        for columns_kind in COLUMNS_KINDS
        for thread_count in _thread_counts()
    ]
