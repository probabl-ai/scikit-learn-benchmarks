from __future__ import annotations

from pydantic import Field, model_validator

from .base import BaseCase, JsonDict, Section


class Algorithm(Section):
    estimator: str
    estimator_params: JsonDict = Field(default_factory=dict)


class Data(Section):
    source: str | None = None
    dataset: str | None = None
    id: int | str | None = None
    generation_kwargs: JsonDict | None = None
    split_kwargs: JsonDict = Field(default_factory=dict)
    preprocessing_kind: str | None = None
    preprocessing_kwargs: JsonDict = Field(default_factory=dict)
    order: str | None = None
    dtype: str | None = None
    x_train: JsonDict | None = None
    x_test: JsonDict | None = None
    y_train: JsonDict | None = None
    y_test: JsonDict | None = None

    @model_validator(mode="after")
    def _check_synthetic_exclusive_options(self) -> "Data":
        if self.generation_kwargs is not None:
            if self.split_kwargs:
                raise ValueError(
                    "split_kwargs is not allowed together with generation_kwargs: "
                    "synthetic data always uses a fixed 50/50 train/test split."
                )
            if self.preprocessing_kind is not None or self.preprocessing_kwargs:
                raise ValueError(
                    "preprocessing_kind/preprocessing_kwargs is not allowed together "
                    "with generation_kwargs: synthetic data is never preprocessed."
                )
        return self

    def name(self, shortened: bool = False) -> str:
        if self.dataset is not None:
            return self.dataset

        source = self.source
        generation_postfix = "".join(
            f"_{key}_{value}" for key, value in (self.generation_kwargs or {}).items()
        )

        if source == "fetch_openml":
            return f"openml_{self.id}"
        if source is not None and source.startswith("make_"):
            if shortened:
                return source.replace("classification", "clsf").replace(
                    "regression", "regr"
                )
            return f"{source}{generation_postfix}"
        raise ValueError("Unable to get data name")


class Implementation(Section):
    library: str
    device: str | None = None
    data_library: str | None = None
    sklearn_context: JsonDict | None = None
    sklearnex_context: JsonDict | None = None

    @property
    def context(self) -> JsonDict:
        return (
            (self.sklearn_context or {})
            | (self.sklearnex_context or {})
        )

    def is_array_api(self) -> bool:
        return self.context.get("array_api_dispatch", False)


class EstimatorCase(BaseCase):
    algorithm: Algorithm
    data: Data
    implementation: Implementation
    runner_module: str = "sklbench.runners.estimator"

    def name(self, shortened: bool = False, separator: str = " ") -> str:
        name_args = [
            self.implementation.library,
            self.algorithm.estimator,
            self.data.name(shortened=shortened),
        ]
        if self.implementation.device is not None:
            name_args.append(self.implementation.device)
        return separator.join(name_args)
