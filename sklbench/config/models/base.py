from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


JsonDict = dict[str, Any]

class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bench(Section):
    n_runs: int = 10
    time_limit: float = 600
    cpu_affinity: list[int] | None = None
    env: dict[str, str] | None = None
    # Off by default: py-spy can stall the profiled run (it stops the
    # process for each sample and falls further behind), and its profiles
    # have not been worth that machine time.
    py_spy_profiling: bool = False
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
