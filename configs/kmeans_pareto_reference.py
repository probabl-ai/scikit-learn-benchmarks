"""Long reference runs of the k-means Pareto benchmark, 5 seeds: greedy k-means++ then
Lloyd until the centers stop moving (or 1,000 iterations), and e-means for 50 epochs (on
glove_200 with 256 clusters, an e-means run at 20 epochs beat every 1,000-iteration Lloyd
run). The dashboard only uses them for the best known inertia of each (dataset,
n_clusters), so that relative inertias stay stable as results are added.

    ./run.sh skl-nvidia faiss-gpu --config configs/kmeans_pareto_reference.py
"""

import os

from kmeans_pareto import DATASETS, N_CLUSTERS, interleaved_cases

GRIDS = {
    "cupy-fused": [{"init": "k-means++", "max_iter": 1000, "tol": 0.0}],
    "emeans-gpu": [{"n_epochs": 50, "batch_law": "quality"}],
}


def generate_cases() -> list[dict]:
    names = {"skl-nvidia": ["cupy-fused"], "faiss-gpu": ["emeans-gpu"]}.get(
        os.environ.get("PIXI_ENVIRONMENT_NAME"), []
    )
    return interleaved_cases(
        names, DATASETS, N_CLUSTERS, grids=GRIDS, n_runs=5, time_limit=7200
    )
