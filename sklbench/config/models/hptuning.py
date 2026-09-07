from __future__ import annotations

import math

from joblib import cpu_count
from pydantic import Field

from .base import BaseCase, JsonDict, Section
from .estimator import Algorithm, Data, Implementation


class HPTuning(Section):
    param_distributions: JsonDict = Field(default_factory=dict)
    n_iter: int = 14
    cv_n_splits: int = 3
    cv_test_size: float = 0.2
    random_state: int = 42
    # None picks "roc_auc" for classification datasets (data.name()'s
    # n_classes is set) and "r2" otherwise - see `_default_scoring` in the
    # runner.
    scoring: str | None = None
    max_samples: int | None = None
    joblib_backend: str = "loky"
    # Outer: parallel RandomizedSearchCV candidate fits (separate workers,
    # via `joblib_backend`). Inner: threads each candidate fit itself may
    # use (BLAS/OpenMP). None means "auto": outer defaults to
    # round(sqrt(physical_cores)) (see `resolve_outer_n_jobs` below); inner
    # is left to joblib's own oversubscription-avoiding default
    # (cpu_count() // outer_n_jobs), applied by passing this value straight
    # through to `joblib.parallel_config(inner_max_num_threads=...)`.
    outer_n_jobs: int | None = None
    inner_n_jobs: int | None = None


def resolve_outer_n_jobs(hptuning: HPTuning) -> int:
    if hptuning.outer_n_jobs is not None:
        return hptuning.outer_n_jobs
    physical_cores = cpu_count(only_physical_cores=True)
    return max(1, round(math.sqrt(physical_cores)))


class HPTuningCase(BaseCase):
    algorithm: Algorithm
    data: Data
    implementation: Implementation = Field(
        default_factory=lambda: Implementation(library="sklearn")
    )
    hptuning: HPTuning = Field(default_factory=HPTuning)
    runner_module: str = "sklbench.runners.hptuning"

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        return separator.join(
            [
                "hptuning",
                self.algorithm.estimator,
                self.data.name(shortened=shortened),
                f"n_iter_{self.hptuning.n_iter}",
                f"outer_{resolve_outer_n_jobs(self.hptuning)}",
            ]
        )
