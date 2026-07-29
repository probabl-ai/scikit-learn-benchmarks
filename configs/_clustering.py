from typing import Iterable


def clustering_cases(template: str) -> Iterable[dict]:
    return _fast_clustering_cases() if template == "fast" else _test_clustering_cases()


def _test_clustering_cases() -> Iterable[dict]:
    generation_kwargs = {
        "centers": 5,
        "cluster_std": 1.0,
        "n_samples": 2000,
        "n_features": 20,
    }
    yield {
        "bench": {"n_runs": 3},
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": {
                "n_clusters": generation_kwargs["centers"],
                "n_init": 1,
                "max_iter": 30,
                "tol": 0.001,
            },
        },
        "data": {
            "source": "make_blobs",
            "generation_kwargs": generation_kwargs,
        },
    }


def _fast_clustering_cases() -> Iterable[dict]:
    bench = {"n_runs": 5, "time_limit": 10}
    params = {"n_init": 1, "max_iter": 30, "tol": 0.001}

    yield {
        "bench": bench,
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": {**params, "n_clusters": 20},
        },
        "data": {
            "source": "make_blobs",
            "generation_kwargs": {
                "centers": 20,
                "cluster_std": 0.5,
                "n_samples": 500000,
                "n_features": 3,
            },
        },
    }
    yield {
        "bench": bench,
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": {**params, "n_clusters": 10},
        },
        "data": {
            "dataset": "mnist",
            "split_kwargs": {"train_size": 50000, "test_size": None},
            "preprocessing_kwargs": {"normalize": "minmax"},
        },
    }
    yield {
        "bench": bench,
        "algorithm": {
            "estimator": "KMeans",
            "estimator_params": {**params, "n_clusters": 100},
        },
        "data": {
            "dataset": "mnist",
            "split_kwargs": {"train_size": 10000, "test_size": 10000},
            "preprocessing_kwargs": {"normalize": "minmax"},
        },
    }
