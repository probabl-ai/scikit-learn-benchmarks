from __future__ import annotations

from typing import Any

from joblib import cpu_count
from pydantic import BaseModel, ConfigDict, Field


JsonDict = dict[str, Any]

# Above this many physical cores, py-spy's ptrace-based sampling can trigger
# a scheduler-churn feedback loop (see `py_spy_rate` in
# `sklbench.orchestrator.commands`) even on cases with little/no
# sklearn-level parallelism, since unpinned BLAS/OpenMP thread pools alone
# are enough to oversubscribe a large core count.
_PY_SPY_MAX_PHYSICAL_CORES = 16


def _py_spy_profiling_default() -> bool:
    return cpu_count(only_physical_cores=True) <= _PY_SPY_MAX_PHYSICAL_CORES


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bench(Section):
    n_runs: int = 10
    time_limit: float = 600
    cpu_affinity: list[int] | None = None
    env: dict[str, str] | None = None
    py_spy_profiling: bool = Field(default_factory=_py_spy_profiling_default)
    py_spy_native: bool = True
    cprofile_profiling: bool = False
    # Runs each of `n_runs` repeats in its own fresh subprocess instead of
    # looping them inside one subprocess. On a multi-NUMA-node machine, all
    # repeats sharing one process also share that process's one memory-
    # placement/thread-scheduling draw (see `configs/_utils/numa.py` and
    # https://github.com/probabl-ai/scikit-learn-benchmarks/issues/80), so
    # the reported timing can reflect a single lucky/unlucky draw rather
    # than a representative sample. Isolating repeats trades wall-clock
    # time (n_runs fresh interpreter/BLAS startups instead of one) for
    # n_runs independent draws of that noise, which a before/after
    # comparison can then average over instead of being dominated by it.
    subprocess_per_repeat: bool = False


class BaseCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bench: Bench = Field(default_factory=Bench)
    metadata: JsonDict = Field(default_factory=dict)
    runner_module: str | None = None

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        raise NotImplementedError
