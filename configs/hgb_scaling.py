from __future__ import annotations

import os
from math import sqrt
from pathlib import Path

import joblib

from _common import sklearn_implementation, with_implementations

DFT_MAX_ITER = 100
DFT_MAX_FEATURES = 0.5


WORKLOADS = [
    {
        "name": "clf-m",
        "problem": "classification",
        "n_samples": 10_000,
        "n_features": 30,
        "max_leaf_nodes": 31,
    },
    {
        "name": "baseline_clf",
        "problem": "classification",
        "n_samples": 100_000,
        "n_features": 100,
        "max_leaf_nodes": 31,
    },
    {
        "name": "all_features_per_split",
        "problem": "classification",
        "n_samples": 100_000,
        "n_features": 100,
        "max_leaf_nodes": 31,
        "max_features": 1.0,
    },
    {
        "name": "stumps_leaf2",
        "problem": "classification",
        "n_samples": 100_000,
        "n_features": 100,
        "max_leaf_nodes": 2,
    },
    {
        "name": "large_leaf255",
        "problem": "classification",
        "n_samples": 100_000,
        "n_features": 100,
        "max_iter": 50,
        "max_leaf_nodes": 255,
    },
    {
        "name": "huge_leaf1000",
        "problem": "classification",
        "n_samples": 60_000,
        "n_features": 100,
        "max_iter": 20,
        "max_leaf_nodes": 1000,
    },
    {
        "name": "few_features",
        "problem": "classification",
        "n_samples": 100_000,
        "n_features": 10,
        "max_leaf_nodes": 31,
        "max_features": 1.0,
    },
    {
        "name": "many_features",
        "problem": "classification",
        "n_samples": 30_000,
        "n_features": 1000,
        "max_iter": 50,
        "max_leaf_nodes": 31,
        "max_features": 0.1,
    },
    {
        "name": "baseline_reg",
        "problem": "regression",
        "n_samples": 100_000,
        "n_features": 100,
        "max_leaf_nodes": 31,
    },
]

WORKLOADS = WORKLOADS[:1]


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


def _case(workload: dict, thread_count: int) -> dict:
    is_classifier = workload["problem"] == "classification"
    estimator = (
        "HistGradientBoostingClassifier"
        if is_classifier
        else "HistGradientBoostingRegressor"
    )
    source = "make_classification" if is_classifier else "make_regression"
    n_features = workload["n_features"]
    default_n_informative = round(sqrt(n_features))
    generation_kwargs = {
        "n_samples": workload["n_samples"],
        "n_features": n_features,
        "n_informative": workload.get("n_informative", default_n_informative),
        "random_state": 0,
    }
    if is_classifier:
        generation_kwargs.update({"n_classes": 2, "n_redundant": 0})

    return {
        "bench": {
            "n_runs": 2,
            "taskset": taskset_for_physical_cores(thread_count),
        },
        "algorithm": {
            "estimator": estimator,
            "estimator_params": {
                "max_iter": workload.get("max_iter", DFT_MAX_ITER),
                "max_leaf_nodes": workload["max_leaf_nodes"],
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


def hgb_scaling_cases() -> list[dict]:
    return [
        _case(workload, thread_count)
        for workload in WORKLOADS
        for thread_count in _thread_counts()
    ]


def generate_cases() -> list[dict]:
    return with_implementations(hgb_scaling_cases(), sklearn_implementation())
