import importlib
import inspect
import logging
from typing import Any

from sklearn.base import BaseEstimator

from ...config import Implementation

logger = logging.getLogger(__name__)

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
