"""Shape x hyperparameter grid of
https://github.com/scikit-learn/scikit-learn/pull/34935#issuecomment-5950111152
(source: https://github.com/ogrisel/scikit-learn/pull/30), with and without
OpenMP active wait, with repeats to tell the outliers it reports from noise.

Meant to run on sklearn-dev builds (the PR branch vs `main`). The original
sweep ran on llvm-openmp with `KMP_BLOCKTIME` 0 and 200: on Linux, use
`sklearn-dev-libomp` to get the same runtime.
"""

from configs.hgb_scalability import _with_thread_count
from _utils.scaling import get_n_cores_list


N_RUNS = 10

SHAPES = [
    ("tiny_1k_x_10", 1_000, 10),
    ("small_5k_x_20", 5_000, 20),
    ("small_wide_2k_x_128", 2_000, 128),
    ("medium_20k_x_20", 20_000, 20),
    ("medium_wide_20k_x_100", 20_000, 100),
    ("medium_50k_x_50", 50_000, 50),
]

HP_CONFIGS = [
    {"hp_name": "tiny_stumps", "max_iter": 10, "max_leaf_nodes": 4, "learning_rate": 0.5},
    {"hp_name": "fast_medium", "max_iter": 30, "max_leaf_nodes": 15, "learning_rate": 0.2},
    {"hp_name": "more_trees", "max_iter": 80, "max_leaf_nodes": 31, "learning_rate": 0.08},
    {"hp_name": "wide_boosted", "max_iter": 80, "max_leaf_nodes": 63, "learning_rate": 0.1},
]

# Both runtimes' knobs are set so the same case toggles active wait on libgomp
# and llvm-openmp builds alike. These are also the values the PR's
# `_openmp_uses_active_wait` and `sklbench.reporting.envs.has_active_wait`
# read back.
WAIT_ENVS = {
    "passive": {"KMP_BLOCKTIME": "0", "GOMP_SPINCOUNT": "1"},
    "active": {"KMP_BLOCKTIME": "200ms", "GOMP_SPINCOUNT": "300000"},
}


def _case(shape: tuple[str, int, int], hp: dict, wait: str, thread_count: int) -> dict:
    shape_name, n_samples, n_features = shape
    name = f"{shape_name}-{hp['hp_name']}"
    case = _with_thread_count({
        "implementation": {
            "library": "sklearn",
        },
        "metadata": {
            "benchmark_type": "scaling",
            "task": "classification",
            "name": name,
        },
        "algorithm": {
            "estimator": "HistGradientBoostingClassifier",
            "estimator_params": {
                "max_iter": hp["max_iter"],
                "max_leaf_nodes": hp["max_leaf_nodes"],
                "learning_rate": hp["learning_rate"],
                "max_bins": 255,
                "early_stopping": False,
                "random_state": 0,
            },
        },
        "data": {
            "id": name,
            "source": "make_classification",
            "generation_kwargs": {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_informative": max(2, n_features // 2),
                "n_redundant": 0,
                "n_repeated": 0,
                "n_classes": 2,
                "n_clusters_per_class": 2,
                "random_state": 0,
            },
            "dtype": "float32",
        },
    }, thread_count)
    case["bench"]["n_runs"] = N_RUNS
    case["bench"]["env"].update(WAIT_ENVS[wait])
    return case


def generate_cases() -> list[dict]:
    return [
        _case(shape, hp, wait, thread_count)
        for shape in SHAPES
        for hp in HP_CONFIGS
        for wait in WAIT_ENVS
        for thread_count in get_n_cores_list()
    ]
