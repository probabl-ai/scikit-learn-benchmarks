from math import sqrt

from _scaling import get_n_cores_list, taskset_for_physical_cores


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
        "columns": columns_kind,
    }
    if is_classifier:
        generation_kwargs.update({"n_classes": 2, "n_redundant": 0})

    return {
        "bench": {
            "n_runs": 2,
            "taskset": taskset_for_physical_cores(thread_count, with_siblings=False),
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
        for thread_count in get_n_cores_list()
    ]
