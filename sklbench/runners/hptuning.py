from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.parallel import Parallel, delayed

from ..config import HPTuningCase
from .datasets import load_raw_data
from .datasets.preprocessing import PREPROCESSORS
from .estimator.loading import get_context, get_estimator

logger = logging.getLogger(__name__)


def _build_pipeline(case: HPTuningCase) -> Pipeline:
    estimator_class = get_estimator(case.implementation.library, case.algorithm.estimator)
    estimator = estimator_class(**case.algorithm.estimator_params)
    preprocessing_kind = case.data.preprocessing_kind
    preprocessor = (
        "passthrough" if preprocessing_kind is None
        else PREPROCESSORS[preprocessing_kind]()
    )
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def _subsample(X, y, n_classes: int | None, max_samples: int | None, random_state: int):
    if max_samples is None or len(X) <= max_samples:
        return X, y
    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        random_state=random_state,
        shuffle=True,
        stratify=y if n_classes is not None else None,
    )
    return X, y


def _default_scoring(n_classes: int | None) -> str:
    if n_classes is None:
        return "r2"
    return "roc_auc" if n_classes == 2 else "roc_auc_ovr"


def _silhouette_scorer(estimator, X, y=None) -> float:
    """Clustering cases (KMeans) have no real `y` to score against - the
    clustering loaders return a dummy all-zero one (see e.g.
    `load_road_network_points`) - and no builtin sklearn scorer name covers
    unsupervised clustering, hence this callable instead. Scores in the
    same (preprocessed) space the estimator actually clustered in, not raw
    `X`. Silhouette needs >=2 clusters to be defined; a degenerate
    single-cluster candidate (possible with `n_clusters` close to a CV
    fold's size) scores the worst possible value instead of raising.
    """
    X_transformed = estimator[:-1].transform(X)
    labels = estimator[-1].predict(X_transformed)
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X_transformed, labels)


# `hptuning.scoring` (a plain str, so it round-trips through the JSON case
# file) resolves through this before reaching RandomizedSearchCV: a name in
# here becomes the matching callable, anything else (a builtin sklearn
# scorer name, or None resolved by `_default_scoring`) passes through as-is.
_SPECIAL_SCORERS = {"silhouette": _silhouette_scorer}


def _raw_xy(raw_data: dict):
    """`load_raw_data` returns `{"x", "y"}` for a loaded dataset (the whole
    thing, unsplit) but `{"x_train", "x_test", "y_train", "y_test"}` for
    synthetic data (already split - see `generate_synthetic_data`). This
    runner doesn't use a held-out test set at all (`best_score_` comes from
    `search`'s own CV folds), so for synthetic data only the train half is
    used, same size as `data.generation_kwargs["n_samples"]`.
    """
    if "x" in raw_data:
        return raw_data["x"], raw_data["y"]
    return raw_data["x_train"], raw_data["y_train"]


def run_hptuning(case: HPTuningCase) -> dict:
    raw_data, data_description = load_raw_data(case)
    n_classes = data_description.get("n_classes")
    hptuning = case.hptuning

    raw_x, raw_y = _raw_xy(raw_data)
    X, y = _subsample(
        raw_x,
        np.asarray(raw_y),
        n_classes,
        hptuning.max_samples,
        hptuning.random_state,
    )
    if not hasattr(X, "iloc"):
        # `make_column_selector` (used by the `PREPROCESSORS` builders) requires a
        # DataFrame - synthetic sources return a plain ndarray.
        X = pd.DataFrame(X)

    pipeline = _build_pipeline(case)
    # Keys are full pipeline param paths (e.g. "estimator__C" to tune the
    # model itself, "preprocessor__categorical__max_categories" to tune
    # preprocessing instead/as well) - not auto-prefixed, so a case can mix
    # both or tune preprocessing exclusively (e.g. when the estimator does
    # its own internal tuning, like RidgeCV's alpha).
    param_distributions = hptuning.param_distributions
    scoring = hptuning.scoring or _default_scoring(n_classes)
    scorer = _SPECIAL_SCORERS.get(scoring, scoring)

    with (
        get_context(case.implementation),
        joblib.parallel_config(
            backend=hptuning.joblib_backend,
            inner_max_num_threads=hptuning.inner_max_num_threads,
        ),
    ):
        # Warm up the worker pool so its startup cost isn't attributed to
        # the timed search below.
        Parallel(n_jobs=hptuning.n_jobs)(delayed(lambda: None)() for _ in range(hptuning.n_jobs))

        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=hptuning.n_iter,
            cv=hptuning.cv_n_splits,
            n_jobs=hptuning.n_jobs,
            scoring=scorer,
            error_score="raise",
            random_state=hptuning.random_state,
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
        "scoring": scoring,
        "best_score": float(search.best_score_),
    }


def run_case_to_jsonl(case: HPTuningCase, n_runs: int, output_jsonl: Path):
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for _ in range(n_runs):
            row = run_hptuning(case)
            fp.write(json.dumps(row) + "\n")
            fp.flush()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m sklbench.runners.hptuning")
    parser.add_argument("--case-file", required=True, type=Path)
    parser.add_argument("--n-runs", required=True, type=int)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(name)s - %(message)s",
    )
    with args.case_file.open("r", encoding="utf-8") as fp:
        case = HPTuningCase.model_validate(json.load(fp))
    run_case_to_jsonl(case, args.n_runs, args.output_jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
