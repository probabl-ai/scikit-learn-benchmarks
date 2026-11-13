"""Runner of the k-means Pareto benchmark (``configs/kmeans_pareto.py``): how long a
fit takes, and how good the centers it returns are.

Cases are `EstimatorCase`s with ``runner_module="sklbench.runners.clustering"``. Unlike
`sklbench.runners.estimator`, a case fits on all the rows of its dataset (no split, no
predict), and the timed part only covers ``fit``:

1. load the dataset into host memory as float32 (a CSR matrix for sparse datasets),
   then copy it to the device (``implementation.data_library``: ``cupy`` on CUDA, or
   ``torch`` on ``implementation.device``). scikit-learn estimators fit on device
   arrays through the array API (``array_api_dispatch=True``);
2. run one untimed warm-up fit with a small budget (see `warmup_params`), so that
   module loading and JIT compilation (CuPy kernels, Triton) stay out of the fit time;
3. for each repeat ``r``, fit with ``random_state=r``, the device synchronized before
   and after, then score the returned centers with the shared evaluator of
   `sklbench.runners.clustering.inertia`.

``data.split_kwargs={"train_size": n}`` fits on a fixed random subset of ``n`` rows.
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from ...config import EstimatorCase
from ..datasets import load_raw_data
from ..estimator.__main__ import (
    _as_jsonable,
    _collect_model_attributes,
    estimator_params_for_repeat,
)
from ..estimator.loading import get_estimator
from .inertia import evaluate_inertia, to_numpy
from .measurement import Device, GPUSampler, peak_rss_bytes, reset_peak_rss, rss_bytes

logger = logging.getLogger(__name__)

# The budget of the warm-up fit, for the parameters a case sets. e-means has no entry:
# its warm-up runs the full budget, because Triton autotunes for the batch shapes of
# the actual fit.
WARMUP_OVERRIDES = {"max_iter": 2, "n_init": 1, "niter": 2, "nredo": 1, "n_epochs": 0.1}
# scikit-learn has no JIT, so a subset of rows pays for imports and first calls.
SKLEARN_WARMUP_ROWS = 100_000
SKLEARN_LIBRARIES = {"sklearn", "sklearn_emeans"}


def warmup_params(case: EstimatorCase, params: dict) -> dict:
    overrides = dict(WARMUP_OVERRIDES)
    if case.algorithm.estimator == "MiniBatchKMeans":
        overrides["max_iter"] = 1  # epochs, not iterations
    return {**params, **{k: v for k, v in overrides.items() if k in params}}


def load_x(case: EstimatorCase):
    raw_data, _ = load_raw_data(case)
    x = raw_data["x"]
    if sp.issparse(x):
        x = sp.csr_matrix(x, dtype=np.float32)
    else:
        x = np.ascontiguousarray(x.to_numpy() if hasattr(x, "to_numpy") else x, dtype=np.float32)
    n_rows = case.data.split_kwargs.get("train_size")
    if n_rows is not None and n_rows < x.shape[0]:
        rng = np.random.default_rng(case.data.split_kwargs.get("random_state", 0))
        x = x[np.sort(rng.choice(x.shape[0], n_rows, replace=False))]
        x = x if sp.issparse(x) else np.ascontiguousarray(x)
    return x


def array_api_context(case: EstimatorCase):
    if case.implementation.data_library and case.implementation.library in SKLEARN_LIBRARIES:
        import sklearn

        return sklearn.config_context(array_api_dispatch=True)
    return contextlib.nullcontext()


def _epochs(estimator, params: dict, n_samples: int) -> float | None:
    """Passes over the full data, over all inits, when known."""
    if getattr(estimator, "n_epochs_", None) is not None:
        return float(estimator.n_epochs_)
    if hasattr(estimator, "n_iter_total_"):  # instrumented scikit-learn KMeans
        return float(estimator.n_iter_total_)
    if hasattr(estimator, "n_steps_"):  # MiniBatchKMeans, EMeans
        batch_size = getattr(estimator, "batch_size_", None) or getattr(
            estimator, "_batch_size", None
        )
        if batch_size is not None:
            return float(estimator.n_steps_ * batch_size / n_samples)
    if params.get("n_init") == 1 and hasattr(estimator, "n_iter_"):
        return float(estimator.n_iter_)
    if params.get("n_epochs") is not None:  # EMeans without n_steps_: no early stopping
        return float(params["n_epochs"])
    return None


def run_case_to_jsonl(case: EstimatorCase, n_runs: int, output_jsonl: Path):
    t0 = perf_counter()
    estimator_class = get_estimator(case.implementation.library, case.algorithm.estimator)
    device = Device(case.implementation.data_library, case.implementation.device)
    import_s = perf_counter() - t0

    t0 = perf_counter()
    X = load_x(case)
    load_s = perf_counter() - t0
    n_samples, n_features = X.shape

    t0 = perf_counter()
    X_device = device.put(X)
    device.sync()
    transfer_s = perf_counter() - t0

    params = dict(case.algorithm.estimator_params)
    t0 = perf_counter()
    rows = X_device.shape[0]
    # On a device, the kernels are compiled for the shapes of the fit, so the warm-up
    # runs on all the rows.
    if case.implementation.library in SKLEARN_LIBRARIES and not device.is_gpu:
        rows = min(rows, max(SKLEARN_WARMUP_ROWS, 4 * params.get("n_clusters", 8)))
    with array_api_context(case):
        estimator_class(
            **estimator_params_for_repeat(estimator_class, warmup_params(case, params), 0)
        ).fit(X_device[:rows])
    device.sync()
    warmup_s = perf_counter() - t0

    setup_ms = {
        "import": 1000 * import_s,
        "load": 1000 * load_s,
        "transfer": 1000 * transfer_s,
        "warmup": 1000 * warmup_s,
    }
    # FAISS on a GPU keeps the data on the host, so this keys on the device.
    sampler = GPUSampler() if case.implementation.device == "cuda" else None
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for repeat in range(n_runs):
            repeat_params = estimator_params_for_repeat(estimator_class, params, repeat)
            estimator = estimator_class(**repeat_params)
            gc.collect()
            device.release()
            device.sync()
            measures = {}
            rss_before = rss_bytes()
            peak_reset = reset_peak_rss()
            if sampler is not None:
                sampler.start()
            with array_api_context(case):
                t0 = perf_counter()
                estimator.fit(X_device)
                device.sync()
                fit_ms = 1000 * (perf_counter() - t0)
            if sampler is not None:
                measures.update(sampler.stop())
            elif peak_reset and rss_before is not None and not device.is_gpu:
                measures["peak_memory_bytes"] = max(peak_rss_bytes() - rss_before, 0)

            centers = to_numpy(estimator.cluster_centers_)
            if centers.shape != (repeat_params.get("n_clusters", 8), n_features):
                raise ValueError(f"centers of unexpected shape {centers.shape}")
            # Apple GPUs have no float64.
            score = evaluate_inertia(X_device if device.is_cuda else X, centers)
            library_inertia = getattr(estimator, "inertia_", None)
            attributes = _collect_model_attributes(estimator)
            attributes["epochs"] = _epochs(estimator, repeat_params, n_samples)
            row = {
                "data_desc": {
                    "fit": {"samples": n_samples, "features": n_features, "dtype": "float32"}
                },
                "time_ms": {"fit": fit_ms},
                "metrics": {
                    "fit": {
                        "inertia_per_sample": score.inertia,
                        "n_empty_clusters": score.n_empty,
                        "n_nonfinite_centers": score.n_nonfinite,
                        "library_inertia_per_sample": (
                            None
                            if library_inertia is None
                            else float(to_numpy(library_inertia)) / n_samples
                        ),
                    }
                },
                "attributes": attributes,
                "measures": measures,
                "setup_ms": setup_ms,
                "random_state": {"estimator": repeat_params.get("random_state")},
                "threads": os.environ.get("OMP_NUM_THREADS"),
            }
            fp.write(json.dumps(row, default=_as_jsonable) + "\n")
            fp.flush()
            print(
                f"repeat {repeat}: fit {fit_ms / 1000:.3f}s, inertia {score.inertia:.6g}, "
                f"n_iter {attributes.get('n_iter')}, empty {score.n_empty}",
                file=sys.stderr,
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m sklbench.runners.clustering")
    parser.add_argument("--case-file", required=True, type=Path)
    parser.add_argument("--n-runs", required=True, type=int)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--cprofile-output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(name)s - %(message)s")
    with args.case_file.open("r", encoding="utf-8") as fp:
        case = EstimatorCase.model_validate(json.load(fp))
    if args.cprofile_output is not None:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
        try:
            run_case_to_jsonl(case, args.n_runs, args.output_jsonl)
        finally:
            profiler.disable()
            profiler.dump_stats(str(args.cprofile_output))
    else:
        run_case_to_jsonl(case, args.n_runs, args.output_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
