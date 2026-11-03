"""
The cases of `gbdt_pareto.py` with every library on a single thread, to
compare serial speed: a hyperparameter search parallelizes over candidates,
so each fit runs on one core.

Single-thread fits are slow, so run this config in many processes at once,
each pinned to its own core: GBDT_PARETO_SHARD="i/n" keeps the i-th of n
shards (from 0). The shards are balanced by a rough cost estimate, and they
filter the cases without changing them.
"""
import os

import configs.gbdt_pareto as gbdt_pareto

THREAD_PARAMS = {
    "sklearn": {},
    "xgboost": {"n_jobs": 1},
    "lightgbm": {"n_jobs": 1},
    "catboost": {"thread_count": 1},
}
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# Single-thread fits of the largest settings take several times longer than
# the multithreaded ones the base configs' time limits are sized for.
TIME_LIMIT_FACTOR = 6
# Approximate training rows and classes (1 for regression), only to balance
# the shards.
DATASET_SIZES = {
    "bank_marketing": (36_000, 1), "amazon_employee_access": (26_000, 1), "kick": (58_000, 1),
    "kddcup09_churn": (40_000, 1), "covtype": (465_000, 7), "ames_housing": (1_200, 1),
    "california_housing": (16_500, 1), "medical_charges_nominal": (130_000, 1),
    "year_prediction_msd": (412_000, 1), "fraud": (228_000, 1), "letters": (16_000, 26),
    "susy": (4_500_000, 1), "abalone": (3_300, 1),
}


def _single_thread(case: dict) -> dict:
    library = case["implementation"]["library"]
    algorithm = case["algorithm"]
    return {
        **case,
        "bench": {
            **case["bench"],
            "env": THREAD_ENV,
            "time_limit": case["bench"]["time_limit"] * TIME_LIMIT_FACTOR,
        },
        "algorithm": {
            **algorithm,
            "estimator_params": {**algorithm["estimator_params"], **THREAD_PARAMS[library]},
        },
    }


def _cost(case: dict) -> float:
    params = case["algorithm"]["estimator_params"]
    n_trees = next(params[k] for k in ("max_iter", "n_estimators", "iterations") if k in params)
    n_leaves = next(params[k] for k in ("max_leaf_nodes", "max_leaves", "num_leaves") if k in params)
    n_rows, n_classes = DATASET_SIZES[case["data"]["dataset"]]
    return n_rows * n_trees * n_leaves * n_classes * case["bench"]["n_runs"]


def _shard(cases: list[dict], index: int, count: int) -> list[dict]:
    """Longest-processing-time-first assignment: each case, most expensive
    first, goes to the least loaded shard."""
    loads = [0.0] * count
    assignment = [[] for _ in range(count)]
    for position in sorted(range(len(cases)), key=lambda i: (-_cost(cases[i]), i)):
        target = min(range(count), key=lambda j: (loads[j], j))
        loads[target] += _cost(cases[position])
        assignment[target].append(position)
    return [cases[i] for i in sorted(assignment[index])]


def generate_cases() -> list[dict]:
    cases = [_single_thread(case) for case in gbdt_pareto.generate_cases()]
    shard = os.environ.get("GBDT_PARETO_SHARD")
    if shard is not None:
        index, count = (int(x) for x in shard.split("/"))
        cases = _shard(cases, index, count)
    return cases
