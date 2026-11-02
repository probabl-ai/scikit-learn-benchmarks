from __future__ import annotations

from pydantic import Field

from .base import BaseCase, JsonDict, Section
from .estimator import Algorithm, Data, Implementation


class HPTuning(Section):
    # Each candidate maps pipeline param paths (e.g. "estimator__C") to one
    # value. Every candidate is fit on `n_splits` random train/test splits.
    candidates: list[JsonDict] = Field(default_factory=lambda: [{}])
    n_splits: int = 3
    test_size: float = 0.2
    random_state: int = 42
    # None picks "roc_auc" for classification datasets (data.name()'s
    # n_classes is set) and "r2" otherwise - see `_default_scoring` in the
    # runner.
    scoring: str | None = None
    max_samples: int | None = None
    joblib_backend: str = "loky"
    # Outer: parallel candidate fits (separate workers, via
    # `joblib_backend`). Inner: threads each fit may use (BLAS/OpenMP),
    # passed to `joblib.parallel_config(inner_max_num_threads=...)`; None
    # keeps joblib's default (cpu_count() // n_jobs).
    n_jobs: int = 1
    inner_max_num_threads: int | None = None


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
                f"k{len(self.hptuning.candidates)}_splits{self.hptuning.n_splits}",
                f"n_jobs{(self.hptuning.n_jobs)}",
            ]
        )
