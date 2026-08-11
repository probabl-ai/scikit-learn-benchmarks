from contextlib import contextmanager
import importlib
import inspect
import logging
from typing import Any

from sklearn.base import BaseEstimator

from ...config import Implementation

logger = logging.getLogger(__name__)

# Emitted by sklearnex's own dispatcher (`sklearnex._utils.PatchingConditionsChain
# .write_log`, via `sklearnex._device_offload.dispatch`) on every patched method
# call. This is the only reliable oneDAL-vs-fallback signal for estimators whose
# accelerated path never sets a detectable attribute on the estimator instance -
# e.g. LogisticRegression on CPU, which routes through daal4py's
# `logistic_regression_path` monkeypatch and never touches `_onedal_estimator`.
_SKLEARNEX_ONEDAL_LOG_MARKER = "running accelerated version on"
_SKLEARNEX_FALLBACK_LOG_MARKER = "fallback to original Scikit-learn"


class _ListLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def capture_sklearnex_dispatch_log():
    """Capture sklearnex's dispatch decision log lines for the duration of the
    `with` block. Yields the (live-appended) list of captured messages."""
    sklearnex_logger = logging.getLogger("sklearnex")
    handler = _ListLogHandler()
    previous_level = sklearnex_logger.level
    previous_propagate = sklearnex_logger.propagate
    sklearnex_logger.addHandler(handler)
    sklearnex_logger.setLevel(logging.INFO)
    sklearnex_logger.propagate = False
    try:
        yield handler.messages
    finally:
        sklearnex_logger.removeHandler(handler)
        sklearnex_logger.setLevel(previous_level)
        sklearnex_logger.propagate = previous_propagate


def sklearnex_used_onedal(messages: list[str]) -> bool | None:
    """Whether sklearnex actually dispatched to oneDAL, based on captured
    dispatch log lines. `None` when no dispatch decision was captured at all
    (unexpected - e.g. logging got reconfigured elsewhere)."""
    if not messages:
        return None
    if any(_SKLEARNEX_FALLBACK_LOG_MARKER in message for message in messages):
        return False
    return any(_SKLEARNEX_ONEDAL_LOG_MARKER in message for message in messages)

ModuleContentMap = dict[str, list[Any]]


def get_module_members(
    module_names_chain: list | str,
) -> tuple[ModuleContentMap, ModuleContentMap]:
    def get_module_name(module_names_chain: list[str]) -> str:
        name = module_names_chain[0]
        for subname in module_names_chain[1:]:
            name += "." + subname
        return name

    def merge_maps(
        first_map: ModuleContentMap, second_map: ModuleContentMap
    ) -> ModuleContentMap:
        output = dict()
        all_keys = set(first_map.keys()) | set(second_map.keys())
        for key in all_keys:
            if key in first_map and key in second_map:
                output[key] = first_map[key] + second_map[key]
            elif key in first_map:
                output[key] = first_map[key]
            elif key in second_map:
                output[key] = second_map[key]
        return output

    if isinstance(module_names_chain, str):
        module_names_chain = [module_names_chain]
    module_name = get_module_name(module_names_chain)
    classes_map: ModuleContentMap = dict()
    functions_map: ModuleContentMap = dict()

    try:
        module = importlib.__import__(module_name, globals(), locals(), [], 0)
        for subname in module_names_chain[1:]:
            module = getattr(module, subname)
    except ModuleNotFoundError:
        return dict(), dict()

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if name in classes_map and obj not in classes_map[name]:
                classes_map[name].append(obj)
            else:
                classes_map[name] = [obj]
        elif inspect.isfunction(obj):
            if name in functions_map and obj not in functions_map[name]:
                functions_map[name].append(obj)
            else:
                functions_map[name] = [obj]

    if hasattr(module, "__all__"):
        for name in module.__all__:
            sub_classes_map, sub_functions_map = get_module_members(
                module_names_chain + [name]
            )
            classes_map = merge_maps(classes_map, sub_classes_map)
            functions_map = merge_maps(functions_map, sub_functions_map)

    return classes_map, functions_map


TASK_TO_ESTIMATOR_SUFFIXES = {
    "classification": ["Classifier", "LogisticRegression", "SVC"],
    "regression": [
        "Regressor",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SVR",
    ],
    "clustering": ["DBSCAN", "KMeans"],
    "decomposition": ["PCA"],
    "manifold": ["TSNE"],
    "search": ["NearestNeighbors"],
    "utility": ["BasicStatistics", "Covariance"],
}


def estimator_to_task(estimator_name: str) -> str:
    """Maps estimator name to machine learning task based on listed estimator postfixes"""
    for task, postfixes_list in TASK_TO_ESTIMATOR_SUFFIXES.items():
        if any(estimator_name.endswith(postfix) for postfix in postfixes_list):
            return task
    return "unknown"


wrapped_estimators = {
    (
        "sklearn",
        "HistGradientBoostingClassifier",
    ): "instrumented_hgb.HistGradientBoostingClassifier",
    (
        "sklearn",
        "HistGradientBoostingRegressor",
    ): "instrumented_hgb.HistGradientBoostingRegressor",
}


def _get_wrapped_estimator(library_name: str, estimator_name: str):
    wrapped_estimator = wrapped_estimators.get((library_name, estimator_name))
    if wrapped_estimator is None:
        return None

    module_name, class_name = wrapped_estimator.rsplit(".", 1)
    module = importlib.import_module(f".wrappers.{module_name}", package=__package__)
    return getattr(module, class_name)


def get_estimator(library_name: str, estimator_name: str):
    # Public classes can be remapped here to wrappers when sklearn API
    # compatibility needs a small adapter.
    estimator = _get_wrapped_estimator(library_name, estimator_name)
    if estimator is not None:
        return estimator

    classes_map, _ = get_module_members(library_name.split("."))
    if estimator_name not in classes_map:
        raise ValueError(
            f"Unable to find {estimator_name} estimator in {library_name} module."
        )
    if len(classes_map[estimator_name]) != 1:
        logger.debug(
            f'List of estimator with name "{estimator_name}": '
            f"{classes_map[estimator_name]}"
        )
        logger.debug(
            f"Found {len(classes_map[estimator_name])} classes for "
            f'"{estimator_name}" estimator name. '
            f"Using first {classes_map[estimator_name][0]}."
        )
    estimator = classes_map[estimator_name][0]
    if not issubclass(estimator, BaseEstimator):
        logger.info(f"{estimator} estimator is not derived from sklearn's BaseEstimator")
    return estimator


def get_context(implementation: Implementation):
    sklearn_context = implementation.sklearn_context
    sklearnex_context = implementation.sklearnex_context

    if sklearnex_context is not None:
        from sklearnex import config_context

        if sklearn_context is not None:
            logger.info(
                f"Updating sklearnex context {sklearnex_context} "
                f"with sklearn context {sklearn_context}"
            )
            sklearnex_context.update(sklearn_context)
        return config_context(**sklearnex_context)
    elif sklearn_context is not None:
        from sklearn import config_context

        return config_context(**sklearn_context)
    else:
        from contextlib import nullcontext

        return nullcontext()
