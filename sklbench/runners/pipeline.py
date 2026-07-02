from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RandomizedSearchCV, ShuffleSplit
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    SplineTransformer,
    TargetEncoder,
)
from sklearn.utils.parallel import Parallel, delayed

from ..config import PipelineCase

logger = logging.getLogger(__name__)


def _as_jsonable(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _as_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(inner) for inner in value]
    return value


def _load_data(case: PipelineCase):
    X, y = fetch_openml(
        data_id=case.data.openml_data_id,
        as_frame=case.data.as_frame,
        return_X_y=True,
    )
    y = y.astype(np.float32).values
    if case.data.max_samples is not None:
        X = X.iloc[: case.data.max_samples]
        y = y[: case.data.max_samples]
    return X, y


def _array_namespace(case: PipelineCase):
    namespace = case.run.array_api_namespace
    if namespace == "numpy":
        return np

    os.environ["SCIPY_ARRAY_API"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*is not currently supported on the MPS.*",
    )
    set_config(array_api_dispatch=True)
    return import_module(namespace)


def _asarray_transformer(xp, device: str):
    if xp is np:
        return FunctionTransformer(
            np.asarray,
            feature_names_out="one-to-one",
            check_inverse=False,
        )
    return FunctionTransformer(
        partial(xp.asarray, device=device),
        feature_names_out="one-to-one",
        check_inverse=False,
    )


def _make_target_encoder(case: PipelineCase):
    return make_pipeline(
        TargetEncoder(
            target_type="continuous",
            cv=KFold(n_splits=5, shuffle=True, random_state=case.run.random_state),
        ),
        MinMaxScaler(),
    )


def _build_pipeline(case: PipelineCase):
    xp = _array_namespace(case)
    target_encoder = _make_target_encoder(case)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                FeatureUnion(
                    [
                        (
                            "onehot",
                            OneHotEncoder(
                                sparse_output=False,
                                handle_unknown="infrequent_if_exist",
                                max_categories=10,
                                min_frequency=5,
                            ),
                        ),
                        ("target", target_encoder),
                    ]
                ),
                make_column_selector(dtype_include=["category", "string"]),
            ),
            (
                "numeric",
                SplineTransformer(n_knots=10, degree=2, handle_missing="zeros"),
                make_column_selector(dtype_include=["number"]),
            ),
        ]
    )
    return make_pipeline(
        preprocessor,
        FunctionTransformer(
            lambda X: X.astype(np.float32),
            feature_names_out="one-to-one",
            check_inverse=False,
        ),
        _asarray_transformer(xp, case.run.device),
        Nystroem(kernel="poly", degree=2, n_components=300, random_state=42),
        RidgeCV(alphas=np.logspace(-6, 6, 13)),
    )


def _default_param_distributions(case: PipelineCase):
    return {
        "columntransformer__numeric__n_knots": [5, 10, 20, 50],
        "columntransformer__categorical__onehot__max_categories": [2, 5, 10, 20, 50],
        "columntransformer__categorical__onehot__min_frequency": [None, 2, 5, 10, 20],
        "columntransformer__categorical__target": ["drop", _make_target_encoder(case)],
        "nystroem__n_components": [10, 30, 100, 300],
        "nystroem__kernel": ["poly", "rbf"],
        "nystroem__degree": [2, 3],
        "nystroem__gamma": [float(value) for value in np.logspace(-6, 6, 25)],
    }


def _n_jobs_values(case: PipelineCase) -> list[int]:
    if case.run.n_jobs is not None:
        return case.run.n_jobs
    max_workers = case.run.max_n_workers or joblib.cpu_count(only_physical_cores=True)
    max_workers = max(1, int(max_workers))
    return [int(2**power) for power in range(int(np.log2(max_workers)) + 1)]


def run_case_once(case: PipelineCase) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning, message=".*A worker stopped.*")
    warnings.filterwarnings("error", category=RuntimeWarning)

    X, y = _load_data(case)
    cv = ShuffleSplit(
        n_splits=case.run.cv_n_splits,
        test_size=case.run.cv_test_size,
        random_state=case.run.random_state,
    )
    pipeline = _build_pipeline(case)
    param_distributions = (
        case.run.param_distributions
        or _default_param_distributions(case)
    )
    timings = []

    with joblib.parallel_config(backend=case.run.joblib_backend):
        for n_jobs in _n_jobs_values(case):
            Parallel(n_jobs=n_jobs)([delayed(lambda: None)() for _ in range(10)])
            search = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=case.run.n_iter,
                cv=cv,
                n_jobs=n_jobs,
                scoring="r2",
                error_score="raise",
                random_state=case.run.random_state,
            )
            tic = time.time()
            error = None
            if case.run.capture_errors:
                try:
                    search.fit(X, y)
                except Exception:
                    error = traceback.format_exc()
            else:
                search.fit(X, y)

            duration_s = time.time() - tic
            timing = {
                "n_jobs": int(n_jobs),
                "duration_s": duration_s,
                "best_r2": None if error is not None else float(search.best_score_),
                "error": error,
            }
            timings.append(timing)

    successful = [timing for timing in timings if timing["error"] is None]
    best_r2 = max((timing["best_r2"] for timing in successful), default=None)
    total_duration_ms = 1000 * sum(timing["duration_s"] for timing in timings)
    return {
        "data_desc": {
            "tune": {
                "samples": len(X),
                "features": int(getattr(X, "shape", [len(X), 0])[1]),
                "openml_data_id": case.data.openml_data_id,
            }
        },
        "time_ms": {"tune": total_duration_ms},
        "metrics": {"tune": {"best_r2": best_r2}},
        "profiling_metrics": {"tune": {}},
        "attributes": {
            "array_api_namespace": case.run.array_api_namespace,
            "device": case.run.device,
            "joblib_backend": case.run.joblib_backend,
            "timings": timings,
        },
    }


def run_case_to_jsonl(case: PipelineCase, n_runs: int, output_jsonl: Path):
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for _ in range(n_runs):
            row = run_case_once(case)
            fp.write(json.dumps(row, default=_as_jsonable) + "\n")
            fp.flush()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m sklbench.runners.pipeline")
    parser.add_argument("--case-file", required=True, type=Path)
    parser.add_argument("--n-runs", required=True, type=int)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(name)s - %(message)s",
    )
    with args.case_file.open("r", encoding="utf-8") as fp:
        case = PipelineCase.model_validate(json.load(fp))
    run_case_to_jsonl(case, args.n_runs, args.output_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
