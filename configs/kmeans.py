"""KMeans-only matrix, for the fit-phase breakdown recorded by
`sklbench/runners/estimator/wrappers/instrumented_kmeans.py` (init vs Lloyd
iterations).

Starts from `_real_datasets.py`'s KMeans datasets and adds intermediate
dimensions by keeping the first `d` columns of the sift and nytimes_256
embeddings (`data.n_features`, sliced after the dataset cache so all prefixes
share one cache entry). fashion_mnist_784 and road_network_points are only
used at full width: the first pixels of fashion_mnist are the mostly-blank
image border, and road_network_points only has 3 columns.

Every workload runs in float32 and float64, with C-ordered input only: KMeans
copies its input to C order anyway, so F-ordered input only adds the cost of
that copy.

Workloads with up to 100 clusters run a fixed 100 Lloyd iterations (`tol=0`,
`max_iter=100`): with the default `tol`, their iteration count varied up to 5x
between repeats, and so did their fit time. Workloads with more clusters keep
the default stopping criterion: on the k=1000 ones (~50 samples per cluster)
the labels stop changing (strict convergence, which `tol=0` does not disable)
after 35-95 iterations, and their fit time is mostly k-means++ init anyway.

Train sizes are subsampled so most float32 fits take roughly 2-15s on a
laptop, plus two bigger workloads (1M samples with k=100, 100k samples with
k=3000) at ~20-25s. The whole matrix (54 cases with the plain sklearn
implementation) takes about 35 minutes there.

On a multi-socket machine, every case is pinned to socket 0 with one OpenMP
thread per physical core. With OpenBLAS, the thread count is still capped at
64 (see `_real_datasets.KMEANS_BENCH`). MKL has no such limit.
"""
import numpy  # noqa: F401, loads the BLAS that threadpoolctl inspects
from threadpoolctl import threadpool_info

from _real_datasets import KMEANS_BENCH
from _utils.implementations import implementations_for_pixi_env
from _utils.numa import cpu_affinity_for_socket


DTYPES = ["float32", "float64"]

# (dataset, n_features, n_clusters, train_size). n_features=None keeps every
# column; train_size=None keeps the dataset's default split.
WORKLOADS = [
    *[("sift", d, 100, 300_000) for d in [16, 32, 64, 128]],
    *[("sift", d, 1000, 50_000) for d in [16, 32, 64, 128]],
    *[("sift", d, 10, None) for d in [32, 64, 128]],
    *[("nytimes_256", d, 100, 150_000) for d in [32, 64, 128, 256]],
    *[("nytimes_256", d, 10, None) for d in [64, 128, 256]],
    *[("nytimes_256", d, 1000, 50_000) for d in [32, 64, 128, 256]],
    ("fashion_mnist_784", None, 100, None),
    *[("road_network_points", None, k, None) for k in [300, 1000]],
    ("sift", 128, 100, None),
    ("sift", 32, 3000, 100_000),
]

# Above this many clusters, keep the default stopping criterion (see above).
MAX_FIXED_ITER_N_CLUSTERS = 100


def _bench() -> dict:
    bench = {"n_runs": 5, "py_spy_profiling": False}
    socket_cpus = cpu_affinity_for_socket(0)
    if socket_cpus is not None:
        bench["cpu_affinity"] = socket_cpus
    if any(pool["internal_api"] == "mkl" for pool in threadpool_info()):
        if socket_cpus is not None:
            # sklearn's default thread count ignores the affinity mask, so
            # it would otherwise start one thread per logical CPU.
            n_cores = len(cpu_affinity_for_socket(0, with_siblings=False))
            bench["env"] = {"OMP_NUM_THREADS": str(n_cores)}
    else:
        bench.update(KMEANS_BENCH)
    return bench


def _case(implem: dict, workload: tuple, dtype: str) -> dict:
    dataset, n_features, n_clusters, train_size = workload
    data = {"dataset": dataset, "dtype": dtype, "order": "C"}
    if n_features is not None:
        data["n_features"] = n_features
    if train_size is not None:
        data["split_kwargs"] = {"train_size": train_size, "test_size": 1000}
    estimator_params = {"n_clusters": n_clusters, "n_init": 1}
    if n_clusters <= MAX_FIXED_ITER_N_CLUSTERS:
        estimator_params |= {"tol": 0, "max_iter": 100}
    return {
        "bench": _bench(),
        "implementation": implem,
        "metadata": {"task": "clustering"},
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": estimator_params,
        },
        "data": data,
    }


def generate_cases() -> list[dict]:
    # sklearn's KMeans has no Array API support, so only plain CPU
    # implementations (sklearn, and sklearnex under `intel`).
    implementations = [
        implem
        for implem in implementations_for_pixi_env()
        if implem.get("data_library") is None and implem.get("device") in (None, "cpu")
    ]
    cases = []
    for implem in implementations:
        for dtype in DTYPES:
            cases += [_case(implem, w, dtype) for w in WORKLOADS]
    return cases
