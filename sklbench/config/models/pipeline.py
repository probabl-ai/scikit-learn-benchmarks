from __future__ import annotations

from pydantic import Field

from .base import BaseCase, JsonDict, Section


class PipelineData(Section):
    openml_data_id: int = 42165
    as_frame: bool = True
    max_samples: int | None = None

    def name(self, shortened: bool = False) -> str:
        return f"openml_{self.openml_data_id}"


class PipelineRun(Section):
    array_api_namespace: str = "numpy"
    device: str = "cpu"
    joblib_backend: str = "loky"
    capture_errors: bool = True
    n_iter: int = 30
    cv_n_splits: int = 3
    cv_test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = 1
    param_distributions: JsonDict = Field(default_factory=dict)


class PipelineCase(BaseCase):
    data: PipelineData = Field(default_factory=PipelineData)
    run: PipelineRun = Field(default_factory=PipelineRun)
    runner_module: str = "sklbench.runners.pipeline"

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        return separator.join(
            [
                "pipeline",
                self.data.name(shortened=shortened),
                self.run.array_api_namespace,
                self.run.device,
                f"n_jobs_{self.run.n_jobs}",
            ]
        )
