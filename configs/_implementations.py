import os
from copy import deepcopy
from itertools import product
from typing import Iterable

from _common import _merge_dicts


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
    "sklearn-pypi": SKLEARN_IMPLEMENTATIONS,
    "sklearn-cf-default": SKLEARN_IMPLEMENTATIONS,
    "sklearn-cf-libgomp-openblas": SKLEARN_IMPLEMENTATIONS,
    "sklearn-cf-libomp-openblas": SKLEARN_IMPLEMENTATIONS,
    "sklearn-cf-libomp-openblas-omp": SKLEARN_IMPLEMENTATIONS,
    "sklearn-cf-mkl": SKLEARN_IMPLEMENTATIONS,
    "sklearn-dev": SKLEARN_IMPLEMENTATIONS,
    "sklearn-dev-libomp": SKLEARN_IMPLEMENTATIONS,
    "skl-cpu": ARRAY_API_CPU_IMPLEMENTATIONS,
    "skl-intel": ARRAY_API_INTEL_IMPLEMENTATIONS,
    "skl-nvidia": ARRAY_API_NVIDIA_IMPLEMENTATIONS,
    "intel": [SKLEARNEX_CPU_IMPLEMENTATION, SKLEARNEX_GPU_IMPLEMENTATION],
}


def implementations_for_pixi_env() -> list[dict]:
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

    return deepcopy(implementations)
