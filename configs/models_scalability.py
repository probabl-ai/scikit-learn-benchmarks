"""
Thread-count scaling study for a small, representative slice of models, each
paired with one real dataset from `real_datasets.py` (real data, not
synthetic like `all_models_scaling.py`'s thread-scaling sweep) - one linear
regressor, one linear classifier, two different tree-ensemble classifiers,
and one clustering estimator:

- Ridge on year_prediction_msd (regression)
- LogisticRegression on covtype (classification, Nystroem-preprocessed)
- ExtraTreesClassifier on susy (classification)
- RandomForestClassifier on fraud (classification)
- KMeans on fashion_mnist_784, n_clusters=100 (clustering)

Cases/hyperparameters are pulled straight from `real_datasets.py` (filtered
to these exact (estimator, dataset) pairs) rather than duplicated here, then
swept across thread count (`taskset_for_physical_cores`, same pattern as
`hgb_scaling.py`/`all_models_scaling.py`) and crossed with every plain-CPU
implementation for the active Pixi environment (`sklearn`, and
`sklearnex-cpu` under `intel`) via `implementations_for_pixi_env` - no
array-API and no sklearnex-gpu implementations, since `taskset`-based thread
pinning isn't a meaningful axis for GPU-offloaded/array-API-dispatched work
(same filter as `all_models_scaling.py`).

RandomForestClassifier/ExtraTreesClassifier's `n_jobs` (baked into
`real_datasets.py` as a fixed fraction of *this* machine's core count) is
overridden to -1 here, so joblib parallelism actually follows the swept
thread count via `taskset` affinity. Ridge/LogisticRegression/KMeans don't
take an `n_jobs` estimator param: their thread count instead follows purely
from BLAS respecting the `taskset` affinity. See `_with_scaling_bench` for
the per-estimator `with_siblings`/`OMP_NUM_THREADS` handling - it isn't
uniform across estimators here.

covtype's LogisticRegression case (Nystroem, 100 components) is ~48s/fit at
`real_datasets.py`'s default ~465K-row train split (see its comment there);
crossed with a thread-count sweep x n_runs x implementations that's too
slow, so its train split is downsized to ~150K rows here (~3x smaller) via
`split_kwargs`.
"""
from _common import _merge_dicts
from _implementations import implementations_for_pixi_env
from _scaling import get_n_cores_list, taskset_for_physical_cores
from real_datasets import generate_cases as generate_real_dataset_cases


MODEL_DATASET_PAIRS = {
    "Ridge": "year_prediction_msd",
    "LogisticRegression": "covtype",
    "ExtraTreesClassifier": "susy",
    "RandomForestClassifier": "fraud",
    "KMeans": "fashion_mnist_784",
}

COVTYPE_TRAIN_SIZE = 150_000

TREE_ESTIMATORS = {"RandomForestClassifier", "ExtraTreesClassifier"}


def _is_array_api(implem: dict) -> bool:
    return bool(
        implem.get("sklearn_context", {}).get("array_api_dispatch")
        or implem.get("sklearnex_context", {}).get("array_api_dispatch")
    )


def _base_cases() -> list[dict]:
    cases = []
    for case in generate_real_dataset_cases():
        estimator = case["algorithm"]["estimator"]
        dataset = case["data"]["dataset"]
        if MODEL_DATASET_PAIRS.get(estimator) != dataset:
            continue

        if estimator in TREE_ESTIMATORS:
            case = _merge_dicts(
                case, {"algorithm": { "estimator_params": {
                    "n_jobs": -1,
                    "max_features": 0.5
                }}}
            )
        if dataset == "covtype":
            case = _merge_dicts(
                case, {"data": {"split_kwargs": {"train_size": COVTYPE_TRAIN_SIZE}}}
            )
        cases.append(case)

    missing = MODEL_DATASET_PAIRS.keys() - {c["algorithm"]["estimator"] for c in cases}
    if missing:
        raise ValueError(f"real_datasets.py produced no case for: {sorted(missing)}")
    return cases



def _with_scaling_bench(case: dict, implem: dict, cores_count: int) -> list[dict]:
    """Builds the sweep point(s) for one (case, implementation, cores_count)
    combo.

    `with_siblings` (whether both hyperthread siblings of each selected
    physical core go into the taskset, vs. just one) only matters for the
    tree cases: they're the only ones with an explicit `n_jobs=-1`, which
    resolves via joblib's plain (not `only_physical_cores`-aware)
    `cpu_count()` - so `with_siblings=True` there would let joblib spawn
    ~2x the intended thread count on hyperthreaded hardware. That's exactly
    the axis worth comparing directly for RF/ET, so both variants are
    yielded (tagged via `metadata["with_siblings"]`, since `bench` - where
    the taskset itself lives - is stripped from case identity elsewhere,
    e.g. `gen_hgb_scaling.py`). Every other estimator here (Ridge/
    LogisticRegression/KMeans) has no `n_jobs`, so `with_siblings=True`
    unconditionally, same as `hgb_scaling.py`.

    KMeans is the only estimator here whose hot loop is OpenMP-parallelized
    in Cython (like HistGradientBoosting), so it's the only one that needs
    `OMP_NUM_THREADS` set explicitly to make thread count follow the sweep
    (Ridge/LogisticRegression/RF/ET rely on BLAS/`n_jobs` respecting the
    `taskset` affinity alone, same as `all_models_scaling.py`). sklearn's
    KMeans is also skipped above 128 threads - see `real_datasets.py`'s
    `KMEANS_BENCH` for the OpenBLAS crash this avoids; not applied to
    sklearnex, whose threading isn't OpenMP/OMP_NUM_THREADS-driven.
    """
    is_sklearn = implem["library"] == "sklearn"
    is_kmeans = case["algorithm"]["estimator"] == "KMeans"
    is_tree = case["algorithm"]["estimator"] in TREE_ESTIMATORS

    if is_sklearn and is_kmeans and cores_count > 128:
        return []

    env = {"OMP_NUM_THREADS": str(cores_count)} if is_sklearn and is_kmeans else {}
    with_siblings_options = [True, False] if is_tree else [True]

    if is_tree:
        # At least one tree per logical core (2 per physical core, since
        # `with_siblings=True`'s taskset - and thus joblib's affinity-derived
        # worker count - covers both hyperthread siblings), so every worker
        # has its own tree to build rather than idling: `real_datasets.py`'s
        # n_estimators is sized off *this* machine's total core count, not
        # the swept cores_count, so it'd undershoot at the low end of the
        # sweep.
        case = _merge_dicts(
            case,
            {"algorithm": {"estimator_params": {"n_estimators": max(24, cores_count * 8)}}},
        )

    return [
        {
            **case,
            "implementation": implem,
            "metadata": {
                **case["metadata"],
                "n_cores": cores_count,
                "with_siblings": with_siblings,
            },
            "bench": {
                **case["bench"],
                "n_runs": 1,
                "env": env,
                "taskset": taskset_for_physical_cores(cores_count, with_siblings),
            },
        }
        for with_siblings in with_siblings_options
    ]


def generate_cases() -> list[dict]:
    implementations = [
        implem
        for implem in implementations_for_pixi_env()
        if not _is_array_api(implem)
    ]
    return [
        scaled_case
        for case in _base_cases()
        for implem in implementations
        for cores_count in get_n_cores_list()
        for scaled_case in _with_scaling_bench(case, implem, cores_count)
    ]
