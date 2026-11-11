"""Temporary: the cases of `kmeans_pareto.py` on its 4 cheapest (dataset, n_clusters)
pairs, to look at the dashboard with real results before the full sweep.

    ./run.sh sklearn-dev faiss-cpu --config configs/kmeans_pareto_fast.py
    ./run.sh faiss-gpu cuml skl-nvidia --config configs/kmeans_pareto_fast.py
"""

import os

from kmeans_pareto import env_implementations, interleaved_cases

PAIRS = [
    ("year_prediction_msd_standardized", 256),
    ("road_network_km_x10", 256),
    ("susy", 256),
    ("sift_128", 256),
]


def generate_cases() -> list[dict]:
    names = env_implementations(os.environ.get("PIXI_ENVIRONMENT_NAME"))
    return [case for dataset, k in PAIRS for case in interleaved_cases(names, [dataset], [k])]
