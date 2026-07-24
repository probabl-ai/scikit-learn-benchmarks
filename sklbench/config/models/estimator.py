from __future__ import annotations

from pydantic import Field

from .base import BaseCase, JsonDict, Section


class Algorithm(Section):
    estimator: str
    estimator_params: JsonDict = Field(default_factory=dict)


class Data(Section):
    source: str | None = None
    dataset: str | None = None
    id: int | str | None = None
    generation_kwargs: JsonDict = Field(default_factory=dict)
    dataset_kwargs: JsonDict = Field(default_factory=dict)
    split_kwargs: JsonDict = Field(default_factory=dict)
    preprocessing_kwargs: JsonDict = Field(default_factory=dict)
    order: str | None = None
    dtype: str | None = None

    def name(self, shortened: bool = False) -> str:
        if self.dataset is not None:
            return self.dataset

        source = self.source
        generation_postfix = "".join(
            f"_{key}_{value}" for key, value in self.generation_kwargs.items()
        )
        dataset_postfix = "".join(
            f"_{key}_{value}" for key, value in self.dataset_kwargs.items()
        )

        if source == "fetch_openml":
            return f"openml_{self.id}"
        if source is not None and source.startswith("make_"):
            if shortened:
                return source.replace("classification", "clsf").replace(
                    "regression", "regr"
                )
            return f"{source}{generation_postfix}{dataset_postfix}"
        raise ValueError("Unable to get data name")


class Implementation(Section):
    library: str
    device: str | None = None
    data_library: str | None = None
    sklearn_context: JsonDict | None = None
    sklearnex_context: JsonDict | None = None


class EstimatorCase(BaseCase):
    algorithm: Algorithm
    data: Data
    implementation: Implementation
    runner_module: str = "sklbench.runners.pipeline"

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        name_args = [
            self.implementation.library,
            self.algorithm.estimator,
            self.data.name(shortened=shortened),
        ]
        if self.implementation.device is not None:
            name_args.append(self.implementation.device)
        return separator.join(name_args)
