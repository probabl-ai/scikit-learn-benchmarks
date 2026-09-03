"""
Thread-count scaling study for RandomForestClassifier, mirroring
`hgb_scalability.py`'s pattern (`taskset_for_physical_cores` + a fixed
workload list swept across `get_n_cores_list()`) but for a forest of
independently built trees instead of HGB's sequential boosting.

Half the workloads are synthetic (`make_trees_classification_data`, sized
XS/S/M/L) and half are real datasets already used with RandomForestClassifier
in `real_datasets.py` (amazon_employee_access, bank_marketing, fraud,
covtype), spanning a similar size range. All cases are classification only:
classification/regression don't need separate coverage here, same as
`hgb_scalability.py`'s TASKS restriction.

`n_estimators`/`max_features` are fixed across the thread sweep (unlike
`real_datasets.py`, which scales `n_estimators` with the host's core count)
so forest size doesn't change together with thread count - only `n_jobs`
does. `n_jobs=-1` + `taskset` is the same thread-pinning pattern
`models_scalability.py` uses for RandomForest/ExtraTrees: joblib picks up
the taskset-restricted CPU affinity rather than being told the count
directly.
"""
from math import sqrt

from _scaling import get_n_cores_list, taskset_for_physical_cores


DFT_N_ESTIMATORS = 300
DFT_MAX_FEATURES = 0.3


SYNTHETIC_WORKLOADS = [
    {"name": "XS", "n_samples": 2_000, "n_features": 20},
    {"name": "S", "n_samples": 20_000, "n_features": 50},
    {"name": "M", "n_samples": 100_000, "n_features": 50},
    {"name": "L", "n_samples": 500_000, "n_features": 20},
]

# Same datasets/estimator pairing as `real_datasets.py`, roughly ordered by
# n_samples to line up with the synthetic S/M/L sizes above (33k / 45k / 285k
# / 581k rows respectively).
REAL_WORKLOADS = [
    {"name": "amazon_employee_access", "dataset": "amazon_employee_access"},
    {"name": "bank_marketing", "dataset": "bank_marketing"},
    {"name": "fraud", "dataset": "fraud"},
    {"name": "covtype", "dataset": "covtype"},
]


def _bench(thread_count: int) -> dict:
    return {
        "n_runs": 3,
        "taskset": taskset_for_physical_cores(thread_count, with_siblings=True),
        "py_spy_profiling": False,
    }


def _estimator_params(thread_count: int) -> dict:
    return {
        "n_estimators": DFT_N_ESTIMATORS,
        "max_features": DFT_MAX_FEATURES,
        "n_jobs": thread_count,
    }


def _synthetic_case(workload: dict, thread_count: int) -> dict:
    n_features = workload["n_features"]
    default_n_informative = round(sqrt(n_features))
    generation_kwargs = {
        "n_samples": workload["n_samples"],
        "n_features": n_features,
        "n_informative": workload.get("n_informative", default_n_informative),
        "n_classes": 2,
        "n_redundant": 0,
        "random_state": 0,
        "columns": "mix",
    }

    return {
        "bench": _bench(thread_count),
        "implementation": {"library": "sklearn"},
        "metadata": {
            "benchmark_type": "scaling",
            "task": "classification",
            "name": workload["name"],
            "kind": "synthetic",
        },
        "algorithm": {
            "estimator": "RandomForestClassifier",
            "estimator_params": _estimator_params(thread_count),
        },
        "data": {
            "id": workload["name"],
            "source": "make_trees_classification_data",
            "generation_kwargs": generation_kwargs,
            "order": "C",
        },
    }


def _real_case(workload: dict, thread_count: int) -> dict:
    return {
        "bench": _bench(thread_count),
        "implementation": {"library": "sklearn"},
        "metadata": {
            "benchmark_type": "scaling",
            "task": "classification",
            "name": workload["name"],
            "kind": "real",
        },
        "algorithm": {
            "estimator": "RandomForestClassifier",
            "estimator_params": _estimator_params(thread_count),
        },
        "data": {
            "dataset": workload["dataset"],
            "preprocessing_kind": "trees",
        },
    }


def generate_cases() -> list[dict]:
    cases = []
    for workload in SYNTHETIC_WORKLOADS:
        for thread_count in get_n_cores_list(max_n_cores=128):
            cases.append(_synthetic_case(workload, thread_count))
    for workload in REAL_WORKLOADS:
        for thread_count in get_n_cores_list(max_n_cores=128):
            cases.append(_real_case(workload, thread_count))
    return cases
