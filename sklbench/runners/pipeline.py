from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from functools import partial
from importlib import import_module
from pathlib import Path

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


def load_data(case: PipelineCase):
    X, y = fetch_openml(
        data_id=case.data.openml_data_id,
        as_frame=True,
        return_X_y=True,
    )
    string_cols = X.select_dtypes(include=["object", "string"]).columns
    X[string_cols] = X[string_cols].astype("category")
    y = y.astype(np.float32).values
    return X, y


def _array_namespace(case: PipelineCase):
    namespace = case.run.array_api_namespace
    if namespace == "numpy":
        return np

    # `SCIPY_ARRAY_API` isn't set here: scipy caches it at import time
    # (`scipy._lib._array_api_override`), and this file's own top-level
    # sklearn imports already pulled scipy in by now. Pixi sets it via
    # `activation.env` in every array-API-capable environment instead.
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


def run_pipeline_tunning(case: PipelineCase) -> dict:
    X, y = load_data(case)
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
    n_jobs = case.run.n_jobs

    with joblib.parallel_config(backend=case.run.joblib_backend):
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
        search.fit(X, y)

    duration_s = time.time() - tic
    return {
        "data_desc": {
            "n_samples": len(X),
            "n_features": X.shape[1],
        },
        "duration_s": duration_s,
        "best_r2": float(search.best_score_),
    }


def run_case_to_jsonl(case: PipelineCase, n_runs: int, output_jsonl: Path):
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for _ in range(n_runs):
            row = run_pipeline_tunning(case)
            fp.write(json.dumps(row) + "\n")
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
