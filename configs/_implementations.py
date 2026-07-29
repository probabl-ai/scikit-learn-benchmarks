import os
from copy import deepcopy
from itertools import product
from typing import Iterable, Literal

from _common import _merge_dicts

Workload = Literal["all_models", "array_api"]


def with_implementations(cases: Iterable[dict], implementations: Iterable[dict]):
    return [
        _merge_dicts(case, {"implementation": implementation})
        for case, implementation in product(cases, implementations)
    ]


SKLEARN_IMPLEMENTATIONS = [{"library": "sklearn"}]


ARRAY_API_CPU_IMPLEMENTATIONS = [
    {
        "library": "sklearn",
        "device": "cpu",
        "data_library": "torch",
        "sklearn_context": {"array_api_dispatch": True},
    }
]


ARRAY_API_INTEL_IMPLEMENTATIONS = [
    {
        "library": "sklearn",
        "device": "xpu",
        "data_library": "torch",
        "sklearn_context": {"array_api_dispatch": True},
    },
    {
        "library": "sklearn",
        "device": "gpu",
        "data_library": "dpnp",
        "sklearn_context": {"array_api_dispatch": True},
    },
]


ARRAY_API_NVIDIA_IMPLEMENTATIONS = [
    {
        "library": "sklearn",
        "device": "cuda",
        "data_library": "torch",
        "sklearn_context": {"array_api_dispatch": True},
    },
    {
        "library": "sklearn",
        "device": "cuda",
        "data_library": "cupy",
        "sklearn_context": {"array_api_dispatch": True},
    },
]


SKLEARNEX_CPU_IMPLEMENTATION = {
    "library": "sklearnex",
    "device": "cpu",
    "sklearnex_context": {
        # TODO? allow to measure cases with fallbacks?
        "allow_fallback_to_host": False,
        "allow_sklearn_after_onedal": False,
    },
}
SKLEARNEX_CPU_IMPLEMENTATIONS = [SKLEARNEX_CPU_IMPLEMENTATION]


# Unused for now:
SKLEARNEX_GPU_IMPLEMENTATION = {
    "library": "sklearnex",
    "device": "gpu",
    "data_library": "dpnp",
    "sklearnex_context": {
        "array_api_dispatch": True,
        "allow_fallback_to_host": False,
        "allow_sklearn_after_onedal": False,
    },
}


PIXI_TO_IMPLEMENTATIONS = {
    "sklearn": SKLEARN_IMPLEMENTATIONS,
    "sklearn-conda": SKLEARN_IMPLEMENTATIONS,
    "sklearn-openblas-pthreads": SKLEARN_IMPLEMENTATIONS,
    "sklearn-openblas-openmp": SKLEARN_IMPLEMENTATIONS,
    "sklearn-mkl": SKLEARN_IMPLEMENTATIONS,
    "sklearn-dev": SKLEARN_IMPLEMENTATIONS,
    "skl-cpu": ARRAY_API_CPU_IMPLEMENTATIONS,
    "skl-intel": ARRAY_API_INTEL_IMPLEMENTATIONS,
    "skl-nvidia": ARRAY_API_NVIDIA_IMPLEMENTATIONS,
    "intel": SKLEARNEX_CPU_IMPLEMENTATIONS,
}


WORKLOAD_TO_PIXI_ENVS: dict[Workload, set[str]] = {
    "all_models": {
        "sklearn",
        "sklearn-conda",
        "sklearn-openblas-pthreads",
        "sklearn-openblas-openmp",
        "sklearn-mkl",
        "sklearn-dev",
        "intel",
    },
    "array_api": set(PIXI_TO_IMPLEMENTATIONS),
}


def implementations_for_pixi_env(*, workload: Workload) -> list[dict]:
    pixi_env = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if pixi_env is None:
        raise ValueError(
            "PIXI_ENVIRONMENT_NAME is not set; load this config through "
            "`pixi run -e <env> ...`."
        )

    implementations = PIXI_TO_IMPLEMENTATIONS.get(pixi_env)
    if implementations is None:
        known_envs = ", ".join(sorted(PIXI_TO_IMPLEMENTATIONS))
        raise ValueError(
            f"Unsupported PIXI_ENVIRONMENT_NAME={pixi_env!r}. "
            f"Expected one of: {known_envs}."
        )

    workload_envs = WORKLOAD_TO_PIXI_ENVS[workload]
    if pixi_env not in workload_envs:
        supported_envs = ", ".join(sorted(workload_envs))
        raise ValueError(
            f"Pixi environment {pixi_env!r} does not support the {workload!r} "
            f"workload. Supported environments: {supported_envs}."
        )

    return deepcopy(implementations)
