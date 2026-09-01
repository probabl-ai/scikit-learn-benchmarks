from math import sqrt

from _scaling import get_n_cores_list, has_hybrid_cores, taskset_for_physical_cores
from real_datasets import generate_cases as generate_real_dataset_cases


DFT_MAX_ITER = 100
DFT_MAX_LEAF_NODES = 31
DFT_MAX_FEATURES = 0.5

# TASKS = ["classification", "regression"]
# COLUMNS_KINDS = ["continuous", "mix"]
# It has been found that the matrix TASKS x COLUMNS_KINDS was
# redundant, so just keeping the simple case:
TASKS = ["classification"]
COLUMNS_KINDS = ["continuous"]


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

    return _with_thread_count({
        "implementation": {
            "library": "sklearn",
        },
        "metadata": {
            "benchmark_type": "scaling",
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
            "order": "C",
        },
    }, thread_count)


def _with_thread_count(case: dict, thread_count: int) -> dict:
    """Apply the same thread-scaling bench overrides as `_case` above to a
    case that already carries its own algorithm/data (e.g. from
    `real_datasets.py`), without touching those sections.

    The explicit OMP_NUM_THREADS is still needed even with our joblib fork
    (github.com/cakedev0/joblib@vendor-loky-cpu-affinity-physical, which
    vendors github.com/cakedev0/loky@cpu_affinity_physical) wired into pixi.toml:
    sklearn's `_openmp_helpers.pyx` picks its default OpenMP thread count via
    `joblib.cpu_count(only_physical_cores=True)`, and without OMP_NUM_THREADS
    that defaults to every logical CPU in the taskset (2x oversubscription
    with `with_siblings=True`, since the fix doesn't take effect in ~half the
    pixi envs - see below). Setting it explicitly makes fit time correct
    everywhere regardless of which joblib is active.

    The fork only reaches environments where scikit-learn/joblib come from
    PyPI/source (sklearn-pypi, sklearn-dev, skl-cpu, skl-intel, skl-nvidia,
    default): there, `[pypi-dependencies].joblib` overrides the vendored
    loky's physical-core-under-affinity detection. In environments where
    scikit-learn is a conda-forge package (intel, reporting, and the
    sklearn-cf-* / BLAS-backend-comparison envs), joblib is pulled in
    transitively as a conda package, and pixi refuses to override a
    conda-selected package with a PyPI git dependency of the same name -
    so those keep using upstream joblib's affinity-blind cpu_count().
    """
    env = {
        "OMP_NUM_THREADS": str(thread_count),
        # "OMP_PROC_BIND": "true"
        # OMP_PROC_BIND=true has several issues:
        # - use with parallel processes/threads it can be very detrimental
        #   (e.g. grid search)
        # - on hybrid cores, it can force using bad cores; while the OS would
        #   have otherwise used fast cores preferably (when n_threads < n_cores)
        # But OMP_PROC_BIND=true is also very beneficial on a big box like the GNR
        # when you use all the available cores (and maybe even a bit on hybrid cores)
    }

    return {
        **case,
        "bench": {
            "n_runs": 5,
            "env": env,
            "taskset": taskset_for_physical_cores(thread_count, with_siblings=True),
            "py_spy_profiling": False,
        },
    }


# 5 of `real_datasets.py`'s HGB-tuned datasets, picked to span its size range
# (~26K to 4.5M train rows: small-medium/small-medium/medium-big/medium-big/
# biggest) while also varying whether the data has categorical columns at
# all:
# - amazon_employee_access (26K x 9, all-categorical) and kddcup09_churn
#   (40K x 207, mostly categorical/messy with heavy missingness) are both
#   categorical but stress different things (low-dim/high-cardinality vs.
#   high-dim/NaN-heavy).
# - year_prediction_msd (464K x 90, all-numeric) and covtype (465K x 12, 10
#   numeric + 2 categorical, multiclass) are near-identical in size, so they
#   isolate the effect of categorical columns at the same scale.
# - susy (4.5M x 18, all-numeric) is by far the biggest, an order of
#   magnitude past the medium-big pair.
REAL_SCALING_DATASETS = {
    "ames_housing",
    "amazon_employee_access",
    "kddcup09_churn",
    "year_prediction_msd",
    "covtype",
    "susy",
}


def _real_dataset_cases() -> list[dict]:
    """Reuse a subset of `real_datasets.py`'s tuned HistGradientBoosting*
    cases (real data, realistic hyperparameters) as extra scaling workloads,
    instead of duplicating dataset/hyperparameter choices here.
    `early_stopping=False` is fixed on every one of those cases specifically
    so they have a fixed iteration count, same as the synthetic workloads
    above - needed for a scaling sweep, where a variable iteration count per
    thread count would confound the measurement."""
    hgb_cases = [
        case
        for case in generate_real_dataset_cases()
        if case["algorithm"]["estimator"].startswith("HistGradientBoosting")
        and case["data"]["dataset"] in REAL_SCALING_DATASETS
    ]
    return [
        _with_thread_count(case, thread_count)
        for case in hgb_cases
        for thread_count in get_n_cores_list()
    ]


def generate_cases() -> list[dict]:
    return [
        _case(workload, task, columns_kind, thread_count)
        for workload in WORKLOADS
        for task in TASKS
        for columns_kind in COLUMNS_KINDS
        for thread_count in get_n_cores_list()
    ] + _real_dataset_cases()
