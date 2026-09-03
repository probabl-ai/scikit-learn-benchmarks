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
    taskset: str | int | None = None
    env: dict[str, str] | None = None
    py_spy_profiling: bool = Field(default_factory=_py_spy_profiling_default)
    py_spy_native: bool = True
    cprofile_profiling: bool = False


class BaseCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bench: Bench = Field(default_factory=Bench)
    metadata: JsonDict = Field(default_factory=dict)
    runner_module: str | None = None

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        raise NotImplementedError
