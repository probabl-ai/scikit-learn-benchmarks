"""Shared KMeans matrix of `kmeans_init.py` and `kmeans_lloyd.py`, for the
fit-phase breakdown recorded by
`sklbench/runners/estimator/wrappers/instrumented_kmeans.py` (init vs Lloyd
iterations).

Starts from `_real_datasets.py`'s KMeans datasets and adds intermediate
dimensions by keeping the first `d` columns of the sift and nytimes_256
embeddings (`data.n_features`, sliced after the dataset cache so all prefixes
share one cache entry). fashion_mnist_784 and road_network_points are only
used at full width: the first pixels of fashion_mnist are the mostly-blank
image border, and road_network_points only has 3 columns.

Wide workloads use CIFAR-10 (3072 features), and sparse ones the CSR text
matrices of 20newsgroups and RCV1, which run KMeans' sparse Lloyd and
k-means++ kernels. Each Lloyd thread allocates a k x n_features buffer, i.e.
38 MB per thread at k=100 on RCV1 in float64.

Every workload runs in float32 and float64, with C-ordered input for the dense
ones: KMeans copies its input to C order anyway, so F-ordered input only adds
the cost of that copy.

On a multi-socket machine, every case is pinned to socket 0 and keeps the
libraries' default thread counts, which follow that affinity: one OpenMP thread
per logical CPU of the socket.
"""
from _utils.implementations import implementations_for_pixi_env
from _utils.numa import cpu_affinity_for_socket


DTYPES = ["float32", "float64"]

# (dataset, n_features, n_clusters, train_size). n_features=None keeps every
# column; train_size=None keeps the dataset's default split.
WORKLOADS = [
    # The sift and nytimes_256 prefixes span 16 to 256 dims.
    *[("sift", d, 100, 300_000) for d in [16, 128]],
    *[("sift", d, 1000, 50_000) for d in [16, 128]],
    ("sift", 128, 10, None),
    *[("nytimes_256", d, 100, 150_000) for d in [64, 256]],
    *[("nytimes_256", d, 1000, 50_000) for d in [64, 256]],
    ("nytimes_256", 256, 10, None),
    ("fashion_mnist_784", None, 100, None),
    # 3 features.
    ("road_network_points", None, 1000, None),
    # Bigger: 1M samples, and 3000 clusters.
    ("sift", 128, 100, None),
    ("sift", 32, 3000, 100_000),
    # Wide: CIFAR-10 pixels, 3072 features.
    ("cifar", None, 100, 10_000),
    # Sparse (CSR) text.
    ("20newsgroups", None, 50, None),
    ("rcv1", None, 100, 50_000),
]

# Loaded as CSR matrices, which have no memory order to set.
SPARSE_DATASETS = {"20newsgroups", "rcv1"}


def _bench() -> dict:
    bench = {"n_runs": 5, "py_spy_profiling": False}
    socket_cpus = cpu_affinity_for_socket(0)
    if socket_cpus is not None:
        bench["cpu_affinity"] = socket_cpus
    return bench


def _case(implem: dict, workload: tuple, dtype: str, estimator_params: dict) -> dict:
    dataset, n_features, n_clusters, train_size = workload
    data = {"dataset": dataset, "dtype": dtype}
    if dataset not in SPARSE_DATASETS:
        data["order"] = "C"
    if n_features is not None:
        data["n_features"] = n_features
    if train_size is not None:
        data["split_kwargs"] = {"train_size": train_size, "test_size": 1000}
    return {
        "bench": _bench(),
        "implementation": implem,
        "metadata": {"task": "clustering"},
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": {"n_clusters": n_clusters, "n_init": 1}
            | estimator_params,
        },
        "data": data,
    }


def generate_kmeans_cases(estimator_params: dict) -> list[dict]:
    """Cases of every workload and dtype, with `estimator_params` added to
    `n_clusters` and `n_init=1`."""
    # sklearn's KMeans has no Array API support, so only plain CPU
    # implementations (sklearn, and sklearnex under `intel`).
    implementations = [
        implem
        for implem in implementations_for_pixi_env()
        if implem.get("data_library") is None and implem.get("device") in (None, "cpu")
    ]
    return [
        _case(implem, workload, dtype, estimator_params)
        for implem in implementations
        for dtype in DTYPES
        for workload in WORKLOADS
    ]
