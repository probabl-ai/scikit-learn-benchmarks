from functools import lru_cache

import sklearn

from .models import Implementation, EstimatorCase


def _sklearn_version() -> tuple[int, int]:
    major, minor = sklearn.__version__.split(".")[:2]
    return (int(major), int(minor))


def supported_logistic_regression_solvers(implem: Implementation | dict):
    if isinstance(implem, dict):
        implem = Implementation(**implem)
    if implem.library == "sklearnex":
        # Any other solver silently falls back to stock scikit-learn. On CPU,
        # newton-cg is only dispatched to oneDAL in sklearnex's preview mode.
        return {"newton-cg"} if implem.device == "gpu" else {"lbfgs"}
    elif implem.library == "sklearn":
        if implem.data_library is None:
            # normal sklearn
            return {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
        solvers = {"lbfgs"}
        if _sklearn_version() >= (1, 10):
            # newton-cholesky gained array API support in 1.10.
            solvers.add("newton-cholesky")
        return solvers
    elif implem.library == "cuml":
        # cuML only runs solver="qn"; wrappers map these sklearn names to it.
        return {"lbfgs", "newton-cg", "newton-cholesky", "qn", "sag", "saga", "liblinear"}
    else:
        raise NotImplementedError()


def select_logistic_regression_solver(implem, solvers):
    """First of `solvers` that `implem` supports, else the last one, in which
    case `filter_unsupported_cases` drops the case."""
    allowed = supported_logistic_regression_solvers(implem)
    for solver in solvers:
        if solver in allowed:
            return solver
    return solvers[-1]


def filter_unsupported_cases(cases):
    for case in cases:
        case = EstimatorCase(**case)
        implem = case.implementation
        estimator = case.algorithm.estimator
        if estimator == "LogisticRegression":
            solver = case.algorithm.estimator_params.get('solver', 'lbfgs')
            if solver not in supported_logistic_regression_solvers(implem):
                continue
            yield case
            continue

        if not implem.is_array_api():
            yield case
            continue

        is_sklearnex = implem.library == "sklearnex"
        if estimator == "RidgeClassifier" and is_sklearnex:
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
    "mps": "mps",
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


@lru_cache
def _mps_gpu_available() -> bool:
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return False
    return torch.backends.mps.is_available()


def _gpu_backend_available(backend: str) -> bool:
    # Dispatches by name (rather than a dict of function refs captured at
    # import time) so tests can monkeypatch `_oneapi_gpu_available` /
    # `_nvidia_gpu_available` / `_mps_gpu_available` on this module and have
    # it take effect here.
    if backend == "oneapi":
        return _oneapi_gpu_available()
    if backend == "nvidia":
        return _nvidia_gpu_available()
    if backend == "mps":
        return _mps_gpu_available()
    raise NotImplementedError(backend)


def filter_gpu_cases_if_unavailable(cases):
    """Drop cases whose `implementation.device` targets a GPU backend
    (oneAPI `gpu`/`xpu`, NVIDIA `cuda`, or Apple `mps`) that isn't actually
    present on this machine.

    Implementation selection (`configs/_utils/implementations.py`) is keyed off
    `PIXI_ENVIRONMENT_NAME` alone, not detected hardware, so e.g. running the
    `intel` Pixi environment on a CPU-only host still generates
    `SKLEARNEX_GPU_IMPLEMENTATION` cases. Those fail at runtime with
    `dpctl._sycl_device.SyclDeviceCreationError` (or the NVIDIA equivalent)
    instead of a clean skip - drop them here instead, the same way
    `filter_unsupported_cases` drops cases an
    implementation doesn't actually support.
    """
    for case in cases:
        device = case["implementation"].get("device")
        backend = _GPU_DEVICE_BACKENDS.get(device)
        if backend is not None and not _gpu_backend_available(backend):
            continue
        yield case


# Estimators with a native cuML wrapper. Anything else in the all_models
# matrix (ExtraTrees, LinearRegression, HistGradientBoosting, ...) has no
# cuML implementation and would fail at estimator construction.
_CUML_ESTIMATORS = {
    "LogisticRegression",
    "Ridge",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "KMeans",
}

# The synthetic LogisticRegression cases are generated once per solver name (lbfgs,
# newton-cg, newton-cholesky). The cuML wrapper runs solver `qn` for every name, so we
# drop the two Newton cases. Real datasets set data["dataset"] and are kept. When one of
# them asks for newton-cholesky, scikit-learn runs it and cuML still runs qn.
_DROPPED_REDUNDANT_CUML_LR_SOLVERS = {"newton-cg", "newton-cholesky"}


def is_cuml_supported_case(case: dict | EstimatorCase) -> bool:
    """Whether this case is one the native cuML wrappers can run.

    The check looks at the estimator and, for synthetic LogisticRegression,
    the solver. It does not look at `implementation.library`, so
    `configs/all_models_cuml_only.py` can apply it to a scikit-learn case too.

    Accepts the dicts `filter_cuml_supported_cases_if_needed` sees and the
    `EstimatorCase` objects `all_models.generate_cases()` returns after
    `filter_unsupported_cases`.
    """
    if isinstance(case, dict):
        estimator = case["algorithm"]["estimator"]
        dataset = case["data"].get("dataset")
        solver = case["algorithm"].get("estimator_params", {}).get("solver")
    else:
        estimator = case.algorithm.estimator
        dataset = case.data.dataset
        solver = case.algorithm.estimator_params.get("solver")
    if estimator not in _CUML_ESTIMATORS:
        return False
    if estimator == "LogisticRegression" and dataset is None:
        if solver in _DROPPED_REDUNDANT_CUML_LR_SOLVERS:
            return False
    return True


def filter_cuml_supported_cases_if_needed(cases):
    """Drop cuML cases the native wrappers cannot run.

    Other libraries are unchanged. Pair a scikit-learn run with cuML through
    `configs/all_models_cuml_only.py`, which applies `is_cuml_supported_case`
    to every case.
    """
    for case in cases:
        if case["implementation"].get("library") == "cuml" and not is_cuml_supported_case(case):
            continue
        yield case
