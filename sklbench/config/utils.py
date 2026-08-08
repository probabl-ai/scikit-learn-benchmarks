import hashlib
import json
import random
from functools import lru_cache

import sklearn

from .models import Implementation, EstimatorCase


def deterministic_random_choice(seed: object, choices: list, n: int = 1):
    """Deterministically pick one of `choices` based on the JSON content of `seed`.

    The same `seed` always yields the same choice, regardless of call order or
    process-level random state, so config scripts stay reproducible while still
    varying a parameter across cases without spelling out every combination.
    """
    digest = hashlib.sha256(
        json.dumps(seed, sort_keys=True).encode("utf-8")
    ).digest()
    rng = random.Random(digest)
    if n == 1:
        return rng.choice(choices)
    else:
        return rng.choices(choices, k=n)


def _sklearn_version() -> tuple[int, int]:
    major, minor = sklearn.__version__.split(".")[:2]
    return (int(major), int(minor))


def supported_logistic_regression_solvers(implem: Implementation | dict):
    if isinstance(implem, dict):
        implem = Implementation(**implem)
    if implem.library == "sklearnex":
        return {'lbfgs', 'newton-cg'}
    elif implem.library == "sklearn":
        if implem.data_library is None:
            # normal sklearn
            return {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
        solvers = {"lbfgs"}
        if _sklearn_version() >= (1, 10):
            # newton-cholesky gained array API support in 1.10.
            solvers.add("newton-cholesky")
        return solvers
    else:
        raise NotImplementedError()


def select_logistic_regression_solver(implem, solvers):
    allowed = supported_logistic_regression_solvers(implem)
    for solver in solvers:
        if solver in allowed:
            return solver
    raise ValueError(f"No supported solvers in {solvers}")


def filter_array_api_supported_cases_if_needed(cases):
    for case in cases:
        case = EstimatorCase(**case)
        implem = case.implementation
        if not implem.is_array_api():
            yield case
            continue

        estimator = case.algorithm.estimator
        is_sklearnex = implem.library == "sklearnex"
        if estimator == "LogisticRegression":
            solver = case.algorithm.estimator_params.get('solver', 'lbfgs')
            if solver not in supported_logistic_regression_solvers(implem):
                continue
        elif estimator == "RidgeClassifier" and is_sklearnex:
            continue
        elif estimator in ("Ridge", "RidgeClassifier"):
            solver = case.algorithm.estimator_params.get('solver', 'auto')
            supported_solvers = ('auto',) if is_sklearnex else ('auto', 'svd')
            if solver not in supported_solvers:
                continue
            if case.data.order == "F" and is_sklearnex:
                # supported but way too slow:
                # https://github.com/uxlfoundation/scikit-learn-intelex/issues/3235
                # TODO: remove this if once the fix is released
                continue
        else:
            continue

        yield case


# implementation.device values that require a GPU backend, mapped to the
# detector below that can tell whether that backend actually has a device.
_GPU_DEVICE_BACKENDS = {
    "gpu": "oneapi",
    "xpu": "oneapi",
    "cuda": "nvidia",
}


@lru_cache
def _oneapi_gpu_available() -> bool:
    try:
        import dpctl
    except (ImportError, ModuleNotFoundError):
        return False
    return any(
        str(device.device_type).split(".")[-1] == "gpu"
        for device in dpctl.get_devices()
    )


@lru_cache
def _nvidia_gpu_available() -> bool:
    try:
        import pynvml
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount() > 0
    except pynvml.NVMLError:
        return False


def _gpu_backend_available(backend: str) -> bool:
    # Dispatches by name (rather than a dict of function refs captured at
    # import time) so tests can monkeypatch `_oneapi_gpu_available` /
    # `_nvidia_gpu_available` on this module and have it take effect here.
    if backend == "oneapi":
        return _oneapi_gpu_available()
    if backend == "nvidia":
        return _nvidia_gpu_available()
    raise NotImplementedError(backend)


def filter_gpu_cases_if_unavailable(cases):
    """Drop cases whose `implementation.device` targets a GPU backend
    (oneAPI `gpu`/`xpu`, or NVIDIA `cuda`) that isn't actually present on
    this machine.

    Implementation selection (`configs/_implementations.py`) is keyed off
    `PIXI_ENVIRONMENT_NAME` alone, not detected hardware, so e.g. running the
    `intel` Pixi environment on a CPU-only host still generates
    `SKLEARNEX_GPU_IMPLEMENTATION` cases. Those fail at runtime with
    `dpctl._sycl_device.SyclDeviceCreationError` (or the NVIDIA equivalent)
    instead of a clean skip - drop them here instead, the same way
    `filter_array_api_supported_cases_if_needed` drops cases an
    implementation doesn't actually support.
    """
    for case in cases:
        device = case["implementation"].get("device")
        backend = _GPU_DEVICE_BACKENDS.get(device)
        if backend is not None and not _gpu_backend_available(backend):
            continue
        yield case
