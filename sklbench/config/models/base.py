from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


JsonDict = dict[str, Any]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bench(Section):
    n_runs: int = 10
    time_limit: float = 600
    taskset: str | int | None = None
    py_spy_profiling: bool = False
    flush_cache: bool = False
    gc_collect: bool = False
    cpu_profile: bool = False
    memory_profile: bool = False
    memory_profiling_interval: float = 0.001


class BaseCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bench: Bench = Field(default_factory=Bench)
    metadata: JsonDict = Field(default_factory=dict)

    @property
    def runner(self) -> str:
        raise NotImplementedError

    def json_dict(self) -> JsonDict:
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        raise NotImplementedError
