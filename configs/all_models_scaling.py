"""
Thread-count scaling study for a small, representative slice of
`all_models.py`'s estimator mix: one tree-ensemble classifier
(ExtraTreesClassifier), one tree-ensemble regressor (RandomForestRegressor),
one linear classifier (LogisticRegression), and one linear regressor (Ridge).

Purpose: on many-core/multi-NUMA-node hardware, `all_models.py` cases were
timing out, and it's unclear whether that's sklearnex/sklearn scaling badly
with thread count (real parallelization overhead, e.g. cross-NUMA effects)
or just the workload sizes growing too large on high-core-count machines.
This config isolates the first question by sweeping thread count
(`taskset_for_physical_cores` + matching `n_jobs`, per `hgb_scaling.py`'s
pattern) at a fixed data size, for both `sklearn` and `sklearnex`, so the
resulting fit-time-vs-thread-count curves can be compared directly per
implementation - and crosses that with a *second* axis, data scale, to see
whether the thread-count scaling behavior itself changes with data size.

The tree cases (ExtraTreesClassifier/RandomForestRegressor) reuse
`synthetic_trees.tree_data_shapes(scale)`'s medium (n_features=20) shape, and
the linear cases (LogisticRegression/Ridge) reuse
`synthetic_linear.linear_data_shapes(scale)`'s medium (n_features=20) shape
likewise, so "scale N" here means the same thing it does in those two
configs (note the two families' shape formulas differ from each other, same
as they already do in `synthetic_trees.py`/`synthetic_linear.py`). `tier`
picks which scales are covered, same cumulative convention as the other
configs:
- test: [1]
- fast: [1, 10]
- normal: [1, 10, 100]
- slow (default): [1, 10, 100, 1000]

LogisticRegression uses "lbfgs" for both implementations (rather than
`select_logistic_regression_solver`) so a solver mismatch doesn't get
conflated with a scaling difference.

Implementation set comes from `PIXI_ENVIRONMENT_NAME` via
`implementations_for_pixi_env`, same as `all_models.py` - so comparing
sklearn vs sklearnex means running this config once under each pixi
environment (e.g. `-e sklearn` and `-e intel`) and comparing results, not a
single invocation covering both. Restricted to plain CPU, non-array-API
implementations only (`taskset`-based thread pinning isn't a meaningful axis
for GPU-offloaded/array-API-dispatched work) - so under array-API-only
environments (`skl-cpu`/`skl-intel`/`skl-nvidia`) this produces no cases at
all, and under `intel` it drops the sklearnex-gpu implementation, keeping
only sklearnex-cpu.
"""
from _implementations import implementations_for_pixi_env
from _scaling import get_n_cores_list, taskset_for_physical_cores
from synthetic_linear import linear_data_shapes
from synthetic_trees import tree_data_shapes


def _is_array_api(implem: dict) -> bool:
    return bool(
        implem.get("sklearn_context", {}).get("array_api_dispatch")
        or implem.get("sklearnex_context", {}).get("array_api_dispatch")
    )


TIER_SCALES = {
    "test": [1],
    "fast": [1, 10],
    "normal": [1, 10, 100],
    "slow": [1, 10, 100, 1000],
}

BENCH = {"n_runs": 3, "time_limit": 300}


def _case(
    estimator: str,
    source: str,
    generation_kwargs: dict,
    estimator_params: dict,
    implem: dict,
    thread_count: int,
    order: str | None = None,
    **extra_metadata,
) -> dict:
    data = {
        "source": source,
        "generation_kwargs": generation_kwargs,
    }
    with_siblings = implem['library'] == "sklearnex"
    if order is not None:
        data["order"] = order
    return {
        "bench": {
            **BENCH,
            "taskset": taskset_for_physical_cores(thread_count, with_siblings),
        },
        "implementation": implem,
        "metadata": {"n_cores": thread_count, **extra_metadata},
        "algorithm": {
            "estimator": estimator,
            "estimator_params": estimator_params,
        },
        "data": data,
    }


def _tree_cases(implem: dict, thread_count: int, scale: int) -> list[dict]:
    # tree_data_shapes(scale) == [n_features=1, n_features=20, n_features=500];
    # the medium (n_features=20) shape is the representative one here.
    shape = tree_data_shapes(scale)[1]
    n_estimators = 300
    return [
        _case(
            "ExtraTreesClassifier",
            "make_trees_classification_data",
            {
                **shape,
                "n_classes": 2,
                "n_redundant": 0,
                "columns": "mix",
                "random_state": 0,
            },
            {
                "n_estimators": n_estimators,
                "max_features": 0.3,
                "random_state": 0,
                "n_jobs": -1,
            },
            implem,
            thread_count,
            scale=scale,
        ),
        _case(
            "RandomForestRegressor",
            "make_trees_regression_data",
            {
                **shape,
                "noise": 0.1,
                "columns": "mix",
                "random_state": 0,
            },
            {
                "n_estimators": n_estimators,
                "max_features": 0.3,
                "random_state": 0,
                "n_jobs": -1,
            },
            implem,
            thread_count,
            scale=scale,
        ),
    ]


def _linear_cases(implem: dict, thread_count: int, scale: int) -> list[dict]:
    # linear_data_shapes(scale) == [n_features=2, n_features=20, n_features=200,
    # n_features=2000]; the medium (n_features=20) shape is the representative
    # one here, same choice as `_tree_cases`.
    shape = linear_data_shapes(scale)[1]

    # Ridge has no `n_jobs`, and LogisticRegression's `n_jobs` only parallelizes
    # multi_class="ovr" across classes - a no-op for binary classification.
    # Both rely on the underlying BLAS threading, which respects the CPU
    # affinity set via `taskset` above, so `thread_count` isn't passed as an
    # estimator param here.
    return [
        _case(
            "LogisticRegression",
            "make_classification",
            {
                **shape,
                "n_classes": 2,
                "n_redundant": 0,
                "random_state": 0,
            },
            {"solver": "lbfgs", "max_iter": 200},
            implem,
            thread_count,
            order="C",
            scale=scale,
        ),
        _case(
            "Ridge",
            "make_regression",
            {
                **shape,
                "noise": 0.1,
                "random_state": 0,
            },
            {"alpha": 1.0},
            implem,
            thread_count,
            order="C",
            scale=scale,
        ),
    ]


def generate_cases(tier: str = "slow") -> list[dict]:
    if tier not in TIER_SCALES:
        raise ValueError(
            f"Unsupported tier={tier!r}. Expected one of: "
            f"{', '.join(sorted(TIER_SCALES))}."
        )
    scales = TIER_SCALES[tier]
    implementations = [
        implem for implem in implementations_for_pixi_env()
        if not _is_array_api(implem)
    ]

    cases = []
    for implem in implementations:
        for thread_count in get_n_cores_list():
            for scale in scales:
                cases += _tree_cases(implem, thread_count, scale)
                cases += _linear_cases(implem, thread_count, scale)
    return cases
