import argparse
import cProfile
from contextlib import nullcontext
import gc
import inspect
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from ...config import EstimatorCase
from ..datasets import load_raw_data, preprocess_data
from .loading import (
    capture_sklearnex_dispatch_log,
    estimator_to_task,
    get_context,
    get_estimator,
    sklearnex_used_onedal,
)
from .metrics import get_subset_metrics_of_estimator

logger = logging.getLogger(__name__)


def detected_core_counts() -> dict:
    """Cores joblib sees under the process's current CPU affinity (e.g. a
    benchmark case's `taskset`, applied once for the whole runner
    subprocess). Fixed for the life of the process - call this once per case
    run, not once per timed call: it can't differ between `fit` and
    `predict`, or between repeats of the same case."""
    return {
        "n_detected_physical_cores": joblib.cpu_count(only_physical_cores=True),
        "n_detected_logical_cpus": joblib.cpu_count(only_physical_cores=False),
    }


def _array_like_size(value: Any) -> int | None:
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size
    if callable(size):
        try:
            size = size()
        except TypeError:
            size = None
        if isinstance(size, int):
            return size
        if isinstance(size, tuple):
            return math.prod(size)

    numel = getattr(value, "numel", None)
    if callable(numel):
        return numel()

    shape = getattr(value, "shape", None)
    if shape is not None:
        return math.prod(shape)
    return None


def _array_like_metadata(value: Any) -> dict | None:
    metadata = {}
    shape = getattr(value, "shape", None)
    if shape is not None:
        metadata["shape"] = list(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        metadata["dtype"] = str(dtype)
    return metadata or None


def _is_singleton_vector(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and tuple(shape) == (1,)


def _as_jsonable(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == (1,):
            return _as_jsonable(value.item())
        if value.size <= 16:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        size = _array_like_size(value)
        if size is not None and size > 16:
            return _array_like_metadata(value)
        try:
            jsonable_value = tolist()
        except (TypeError, ValueError, RuntimeError):
            return _array_like_metadata(value)
        if _is_singleton_vector(value) and isinstance(jsonable_value, list):
            return _as_jsonable(jsonable_value[0])
        return _as_jsonable(jsonable_value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _as_jsonable(inner_value)
            for key, inner_value in value.items()
            if _as_jsonable(inner_value) is not None
        }
    if isinstance(value, (list, tuple)):
        if len(value) <= 16:
            return [_as_jsonable(inner_value) for inner_value in value]
        return {"length": len(value)}
    return None


def _collect_model_attributes(estimator) -> dict[str, Any]:
    attributes = {}
    for attribute_name in [
        "n_iter_",
        "solver_",
        "n_features_in_",
        "n_outputs_",
        "n_clusters_",
        "n_components_",
        "oob_score_",
    ]:
        if not hasattr(estimator, attribute_name):
            continue
        value = getattr(estimator, attribute_name)
        if attribute_name == "support_vectors_" and value is not None:
            value = len(value)
        jsonable = _as_jsonable(value)
        if jsonable is not None:
            attributes[attribute_name.rstrip("_")] = jsonable

    # XXX: for sklearnex this is to slow:
    # if hasattr(estimator, "estimators_") and hasattr(estimator.estimators_[0], "tree_"):
    #     attributes["avg_n_leaves"] = statistics.mean(
    #         tree.tree_.n_leaves for tree in estimator.estimators_
    #     )
    #     attributes["avg_max_depth"] = statistics.mean(
    #         tree.tree_.max_depth for tree in estimator.estimators_
    #     )

    for name, value in vars(estimator).items():
        if not name.endswith("_") or name.startswith("_"):
            # not a public attribute
            continue
        attribute_name = name.rstrip("_")
        if attribute_name.rstrip("_") in attributes:
            # already collected
            continue
        if isinstance(value, (int, float, str)):
            attributes[attribute_name] = _as_jsonable(value)

    return attributes


def run_case_once(
    bench_case: EstimatorCase,
    estimator,
    raw_data,
    data_description: dict,
) -> dict:
    task = estimator_to_task(bench_case.algorithm.estimator)
    is_sklearnex = bench_case.implementation.library == "sklearnex"

    times = {}

    # Preprocessed fresh on every repeat, not once up front, so the
    # preprocessing pipeline itself is measured like `fit`/`predict`.
    t0 = perf_counter()
    data, subset_description = preprocess_data(bench_case, raw_data, data_description)
    times["preprocessing"] = 1000 * (perf_counter() - t0)
    X_train, X_test, y_train, y_test = data

    # Encoders (OneHotEncoder/OrdinalEncoder/TargetEncoder, ...) have no
    # `predict` - `transform` is their equivalent inference-time call, timed
    # under the same "predict" key so existing reporting (fit vs. predict/
    # train vs. test) keeps working unchanged.
    inference_method_name = "transform" if task == "encoding" else "predict"

    with (
        capture_sklearnex_dispatch_log()
        if is_sklearnex
        else nullcontext([])
    ) as dispatch_log:
        t0 = perf_counter()
        estimator.fit(X_train, y_train)
        times["fit"] = 1000 * (perf_counter() - t0)

        t0 = perf_counter()
        getattr(estimator, inference_method_name)(X_test)
        times["predict"] = 1000 * (perf_counter() - t0)

        quality_metrics = {
            "fit": get_subset_metrics_of_estimator(
                task, "training", estimator, (X_train, y_train)
            ),
            "predict": get_subset_metrics_of_estimator(
                task, "inference", estimator, (X_test, y_test)
            ),
        }

    data_desc = {
        "preprocessing": dict(subset_description["preprocessing"]),
        "fit": dict(subset_description["x_train"]),
        "predict": dict(subset_description["x_test"]),
    }
    if "n_classes" in subset_description:
        data_desc["preprocessing"].update({"n_classes": subset_description["n_classes"]})
        data_desc["fit"].update({"n_classes": subset_description["n_classes"]})
        data_desc["predict"].update({"n_classes": subset_description["n_classes"]})

    attributes = _collect_model_attributes(estimator)
    if is_sklearnex:
        # sklearnex silently falls back to stock sklearn per-estimator when
        # oneDAL doesn't support the given params/data (e.g. RandomForestClassifier
        # with class_weight="balanced_subsample"), so a "sklearnex" record isn't
        # necessarily measuring oneDAL acceleration. `_onedal_estimator` is set by
        # most estimators' accelerated path but not all - e.g. LogisticRegression
        # on CPU routes through daal4py and never sets it - so prefer sklearnex's
        # own dispatch log when available and fall back to the attribute check.
        used_onedal = sklearnex_used_onedal(dispatch_log)
        if used_onedal is None:
            used_onedal = hasattr(estimator, "_onedal_estimator")
        attributes["has_onedal_estimator"] = used_onedal

    return {
        "data_desc": data_desc,
        "time_ms": times,
        "metrics": quality_metrics,
        "attributes": attributes,
    }


def estimator_params_for_repeat(
    estimator_class, estimator_params: dict, repeat: int
) -> dict:
    params = dict(estimator_params)
    if "random_state" in params:
        return params

    try:
        signature = inspect.signature(estimator_class)
    except (TypeError, ValueError):
        return params
    if "random_state" in signature.parameters:
        params["random_state"] = repeat
    return params


def run_case_to_jsonl(bench_case: EstimatorCase, n_runs: int, output_jsonl: Path):
    library_name = bench_case.implementation.library
    estimator_name = bench_case.algorithm.estimator
    estimator_class = get_estimator(library_name, estimator_name)

    raw_data, data_description = load_raw_data(bench_case)
    estimator_params = dict(bench_case.algorithm.estimator_params)
    detected_cores = detected_core_counts()

    with (
        output_jsonl.open("w", encoding="utf-8") as fp,
        get_context(bench_case.implementation),
    ):
        for repeat in range(n_runs):
            gc.collect()
            repeat_estimator_params = estimator_params_for_repeat(
                estimator_class, estimator_params, repeat
            )
            row = run_case_once(
                bench_case,
                estimator_class(**repeat_estimator_params),
                raw_data,
                data_description,
            )
            row["detected_cores"] = detected_cores
            fp.write(json.dumps(row, default=_as_jsonable) + "\n")
            fp.flush()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m sklbench.runners.estimator")
    parser.add_argument("--case-file", required=True, type=Path)
    parser.add_argument("--n-runs", required=True, type=int)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--cprofile-output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(name)s - %(message)s",
    )
    with args.case_file.open("r", encoding="utf-8") as fp:
        bench_case = EstimatorCase.model_validate(json.load(fp))

    if args.cprofile_output is not None:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            run_case_to_jsonl(bench_case, args.n_runs, args.output_jsonl)
        finally:
            profiler.disable()
            profiler.dump_stats(str(args.cprofile_output))
    else:
        run_case_to_jsonl(bench_case, args.n_runs, args.output_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
