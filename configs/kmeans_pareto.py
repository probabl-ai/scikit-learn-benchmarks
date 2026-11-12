"""k-means Pareto fronts: for a given time budget, which implementation and settings
give the lowest inertia? Port of the v1 sweep of
https://github.com/cakedev0/kmeans-benchmarks.

Each case is one setting of one implementation, fitted 3 times (random_state 0 to 2)
on all the rows of a dataset, by `sklbench.runners.clustering`. Each pixi environment
runs its own implementations:

    ./run.sh sklearn-dev faiss-cpu --config configs/kmeans_pareto.py       # CPU
    ./run.sh faiss-gpu cuml skl-nvidia --config configs/kmeans_pareto.py   # CUDA GPU

The EMeans of cakedev0/scikit-learn#46 runs in a sklearn-dev build of that branch:

    ./run.sh sklearn-dev@cakedev0:<branch> --config configs/kmeans_pareto.py

Other environments get no cases. The ``FastNumpyEmeans`` cases of ``faiss-cpu`` need
its kernel: ``pixi run -e faiss-cpu build-emeans-kernel``.

Budget grids set a tiny ``tol``, so that ``max_iter`` (or the epochs) is the knob
actually swept. A second grid keeps each library's default stopping rule.
"""

from itertools import product
import os

import joblib

from _utils.scaling import cpu_affinity_for_physical_cores

DATASETS = [
    "glove_200",
    "sift_128",
    "clip_512",
    "susy",
    "year_prediction_msd_standardized",
    "road_network_km_x10",
]
N_FEATURES = {
    "glove_200": 200,
    "sift_128": 128,
    "clip_512": 512,
    "susy": 18,
    "year_prediction_msd_standardized": 90,
    "road_network_km_x10": 3,
}
N_CLUSTERS = [256, 1024]

# Above 64 OpenMP threads, scikit-learn's KMeans crashes with OpenBLAS (see
# _real_datasets.py). Bigger machines run on their first 64 physical cores, both SMT
# siblings included.
MAX_THREADS = 64
N_THREADS = min(joblib.cpu_count(only_physical_cores=True), MAX_THREADS)
CPU_AFFINITY = (
    cpu_affinity_for_physical_cores(MAX_THREADS, with_siblings=True)
    if joblib.cpu_count(only_physical_cores=True) > MAX_THREADS
    else None
)

_BUDGET = [3, 10, 30, 100, 300]
_EPOCHS = [0.5, 1, 2, 5, 10]
_FAISS_GRID = [{"niter": [3, 10, 25, 100]}, {"niter": 25, "max_points_per_centroid": 256}]
_EMEANS_CPU_GRID = [{"n_epochs": _EPOCHS}, {"n_epochs": [2, 5], "batch_law": "quality"}]
_EMEANS_DEFAULTS = {"n_epochs": 5, "batch_law": "auto", "batch_size": None, "alpha0": None}
_SKLEARN = {"library": "sklearn"}
_FAISS_CPU = {"library": "faiss", "device": "cpu"}
_CUDA_TORCH = {"library": "faiss", "device": "cuda", "data_library": "torch"}
_CUDA_CUPY = {"device": "cuda", "data_library": "cupy"}

# name: implementation, estimator, default parameters, grids (each a cartesian
# product over the defaults), and optionally the largest number of features it runs on.
IMPLEMENTATIONS = {
    "sklearn-lloyd": {
        "implementation": _SKLEARN,
        "estimator": "KMeans",
        "defaults": {"algorithm": "lloyd", "init": "k-means++", "n_init": 1, "max_iter": 300, "tol": 1e-4},
        "grids": [
            {"init": ["random", "k-means++"], "max_iter": _BUDGET, "tol": 1e-9},
            {"init": ["random", "k-means++"], "n_init": [1, 3]},
        ],
    },
    # Elkan only pays off in low dimension.
    "sklearn-elkan": {
        "implementation": _SKLEARN,
        "estimator": "KMeans",
        "defaults": {"algorithm": "elkan", "init": "k-means++", "n_init": 1, "max_iter": 300, "tol": 1e-4},
        "grids": [{"init": ["random", "k-means++"], "max_iter": _BUDGET, "tol": 1e-9}, {}],
        "max_features": 3,
    },
    # Not in ENV_IMPLEMENTATIONS for now: ~230 s per epoch at batch_size=1024 on susy,
    # from per-step overhead.
    "sklearn-minibatch": {
        "implementation": _SKLEARN,
        "estimator": "MiniBatchKMeans",
        "defaults": {
            "init": "k-means++",
            "n_init": 1,
            "max_iter": 100,
            "batch_size": 1024,
            "tol": 0.0,
            "max_no_improvement": 10,
            "reassignment_ratio": 0.01,
        },
        "grids": [
            {"batch_size": [8192, 65536], "max_iter": [1, 3, 10]},
            {"batch_size": 65536, "max_iter": 30, "max_no_improvement": None},
            {"batch_size": 1024, "max_iter": 1},
        ],
    },
    # Its fit ends with a labels pass on the final centers (about one more epoch).
    "sklearn-emeans": {
        "implementation": _SKLEARN,
        "estimator": "EMeans",
        "defaults": {"init": "random", "n_epochs": 5},
        "grids": [{"n_epochs": _EPOCHS, "init": ["random", "k-means++"]}],
    },
    "faiss-cpu": {
        "implementation": _FAISS_CPU,
        "estimator": "Kmeans",
        "defaults": {"niter": 25, "nredo": 1, "max_points_per_centroid": None},
        "grids": _FAISS_GRID,
    },
    "emeans-cpu": {
        "implementation": _FAISS_CPU,
        "estimator": "NumpyEmeans",
        "defaults": _EMEANS_DEFAULTS,
        "grids": _EMEANS_CPU_GRID,
    },
    "emeans-cpu-cython": {
        "implementation": _FAISS_CPU,
        "estimator": "FastNumpyEmeans",
        "defaults": _EMEANS_DEFAULTS,
        "grids": _EMEANS_CPU_GRID,
    },
    "faiss-gpu": {
        "implementation": {"library": "faiss", "device": "cuda"},
        "estimator": "Kmeans",
        "defaults": {"niter": 25, "nredo": 1, "max_points_per_centroid": None, "gpu": True},
        "grids": _FAISS_GRID,
    },
    "emeans-gpu": {
        "implementation": _CUDA_TORCH,
        "estimator": "QuantizedEmeans",
        "defaults": _EMEANS_DEFAULTS,
        "grids": [
            {"n_epochs": [0.5, 1, 2, 3, 5, 10, 20]},
            {"n_epochs": [2, 5, 10], "batch_law": "quality"},
        ],
    },
    # Reduced precision: BF16 operands.
    "emeans-gpu-bf16": {
        "implementation": _CUDA_TORCH,
        "estimator": "Emeans",
        "defaults": _EMEANS_DEFAULTS,
        "grids": [{"n_epochs": [0.5, 1, 2, 5, 10, 20]}],
    },
    # cuML's "k-means++" is greedy k-means++, "scalable-k-means++" is k-means||.
    "cuml": {
        "implementation": {"library": "cuml", **_CUDA_CUPY},
        "estimator": "KMeans",
        "defaults": {
            "init": "scalable-k-means++",
            "n_init": 1,
            "max_iter": 300,
            "tol": 1e-4,
            "oversampling_factor": 2.0,
        },
        "grids": [
            {"init": ["random", "k-means++", "scalable-k-means++"], "max_iter": _BUDGET, "tol": 1e-9},
            {"init": ["random", "k-means++", "scalable-k-means++"], "n_init": [1, 3]},
        ],
    },
    "cupy": {
        "implementation": {"library": "kmeans_lloyd", **_CUDA_CUPY},
        "estimator": "KMeans",
        "defaults": {"init": "k-means++", "n_init": 1, "max_iter": 300, "tol": 1e-4, "fused": False},
        "grids": [
            {"init": ["random", "k-means++"], "max_iter": _BUDGET, "tol": 1e-9},
            {"init": ["random", "k-means++"]},
        ],
    },
    "cupy-fused": {
        "implementation": {"library": "kmeans_lloyd", **_CUDA_CUPY},
        "estimator": "KMeans",
        "defaults": {"init": "k-means++", "n_init": 1, "max_iter": 300, "tol": 1e-4, "fused": True},
        "grids": [
            {"init": ["random", "k-means++"], "max_iter": _BUDGET, "tol": 1e-9},
            {"init": ["random", "k-means++"], "n_init": [1, 3]},
        ],
    },
}


def _sklearn_has_emeans() -> bool:
    import sklearn.cluster

    return hasattr(sklearn.cluster, "EMeans")


def env_implementations(pixi_env: str | None) -> list[str]:
    """Implementations of `IMPLEMENTATIONS` that run in `pixi_env`."""
    if pixi_env == "sklearn-dev":
        # On a build of cakedev0/scikit-learn#46, only its EMeans runs: the KMeans
        # cases come from main.
        return ["sklearn-emeans"] if _sklearn_has_emeans() else ["sklearn-lloyd", "sklearn-elkan"]
    return {
        "faiss-cpu": ["faiss-cpu", "emeans-cpu", "emeans-cpu-cython"],
        "faiss-gpu": ["faiss-gpu", "emeans-gpu", "emeans-gpu-bf16"],
        "cuml": ["cuml"],
        "skl-nvidia": ["cupy", "cupy-fused"],
    }.get(pixi_env, [])


def settings(name: str, grids: list[dict] | None = None) -> list[dict]:
    """Every setting of implementation `name`: the union of its grids, each a cartesian
    product merged over its defaults, without duplicates."""
    spec = IMPLEMENTATIONS[name]
    points = []
    for grid in spec["grids"] if grids is None else grids:
        keys = list(grid)
        values = [v if isinstance(v, list) else [v] for v in grid.values()]
        for combination in product(*values):
            point = {**spec["defaults"], **dict(zip(keys, combination))}
            if point not in points:
                points.append(point)
    return points


def _case(name, params, dataset, n_clusters, *, n_runs, time_limit, split_kwargs=None):
    spec = IMPLEMENTATIONS[name]
    env = {f"{lib}_NUM_THREADS": str(N_THREADS) for lib in ("OMP", "OPENBLAS", "MKL")}
    if spec["implementation"].get("device") == "cuda":
        env |= {"NVIDIA_TF32_OVERRIDE": "0", "CUPY_TF32": "0"}
    data = {"dataset": dataset, "dtype": "float32"}
    if split_kwargs:
        data["split_kwargs"] = split_kwargs
    return {
        "algorithm": {
            "estimator": spec["estimator"],
            "estimator_params": {"n_clusters": n_clusters, **params},
        },
        "data": data,
        "implementation": spec["implementation"],
        "runner_module": "sklbench.runners.clustering",
        "bench": {"n_runs": n_runs, "time_limit": time_limit, "env": env, "cpu_affinity": CPU_AFFINITY},
    }


def interleaved_cases(
    names: list[str],
    datasets: list[str],
    n_clusters: list[int],
    *,
    grids: dict | None = None,
    n_runs: int = 3,
    time_limit: float = 3600,
    split_kwargs: dict | None = None,
) -> list[dict]:
    """Cases per (dataset, n_clusters), going round-robin over the implementations and
    rotating which one starts each round, so that slow drifts of the machine (thermals,
    a GPU's power cap) spread over all of them."""
    cases = []
    for dataset, k in product(datasets, n_clusters):
        per_name = {
            name: settings(name, (grids or {}).get(name))
            for name in names
            if N_FEATURES.get(dataset, 0) <= IMPLEMENTATIONS[name].get("max_features", float("inf"))
        }
        order = list(per_name)
        for round_ in range(max(map(len, per_name.values()), default=0)):
            shift = round_ % len(order)
            for name in order[shift:] + order[:shift]:
                if round_ < len(per_name[name]):
                    params = per_name[name][round_]
                    cases.append(
                        _case(
                            name, params, dataset, k,
                            n_runs=n_runs, time_limit=time_limit, split_kwargs=split_kwargs,
                        )
                    )
    return cases


def generate_cases() -> list[dict]:
    names = env_implementations(os.environ.get("PIXI_ENVIRONMENT_NAME"))
    return interleaved_cases(names, DATASETS, N_CLUSTERS)
