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

Train sizes are subsampled so every float32 fit takes roughly 2-15s on a
laptop. The whole matrix (50 cases with the plain sklearn implementation)
takes about 35 minutes there.
"""
from _real_datasets import KMEANS_BENCH
from _utils.implementations import implementations_for_pixi_env


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
]


def _case(implem: dict, workload: tuple, dtype: str) -> dict:
    dataset, n_features, n_clusters, train_size = workload
    data = {"dataset": dataset, "dtype": dtype, "order": "C"}
    if n_features is not None:
        data["n_features"] = n_features
    if train_size is not None:
        data["split_kwargs"] = {"train_size": train_size, "test_size": 1000}
    return {
        "bench": {"n_runs": 5, "py_spy_profiling": False, **KMEANS_BENCH},
        "implementation": implem,
        "metadata": {"task": "clustering"},
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": {"n_clusters": n_clusters},
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
