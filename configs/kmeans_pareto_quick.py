"""A few minutes per environment: a handful of settings of every implementation of
`kmeans_pareto.py`, on 100,000-row subsets with 64 clusters: year_prediction_msd_standardized,
and road_network_km_x10 for Elkan (3 features). To check an environment before the full
sweep:

    ./run.sh sklearn-dev faiss-cpu --config configs/kmeans_pareto_quick.py
"""

import os

from kmeans_pareto import IMPLEMENTATIONS, env_implementations, interleaved_cases

_LLOYD = [{"init": ["random", "k-means++"], "max_iter": [2, 5, 20, 100], "tol": 1e-9}]
_EMEANS = [{"n_epochs": [0.5, 1, 2, 5]}]
GRIDS = {
    "sklearn-lloyd": _LLOYD,
    "sklearn-elkan": [{"init": "k-means++", "max_iter": [5, 100], "tol": 1e-9}],
    "sklearn-emeans": _EMEANS,
    "sklearn-contrib-emeans": _EMEANS,
    "faiss-cpu": [{"niter": [2, 5, 20, 100]}],
    "emeans-cpu": _EMEANS,
    "emeans-cpu-cython": _EMEANS,
    "faiss-gpu": [{"niter": [2, 5, 20, 100]}],
    "emeans-gpu": _EMEANS,
    "emeans-gpu-bf16": _EMEANS,
    "cuml": [{"init": ["random", "k-means++", "scalable-k-means++"], "max_iter": [2, 20], "tol": 1e-9}],
    "cupy": _LLOYD,
    "cupy-fused": _LLOYD,
}
assert set(GRIDS) | {"sklearn-minibatch"} == set(IMPLEMENTATIONS)


def generate_cases() -> list[dict]:
    names = env_implementations(os.environ.get("PIXI_ENVIRONMENT_NAME"))
    kwargs = {
        "grids": GRIDS,
        "time_limit": 600,
        "split_kwargs": {"train_size": 100_000, "random_state": 0},
    }
    return interleaved_cases(
        names, ["year_prediction_msd_standardized"], [64], **kwargs
    ) + interleaved_cases(
        [name for name in names if name == "sklearn-elkan"], ["road_network_km_x10"], [64], **kwargs
    )
