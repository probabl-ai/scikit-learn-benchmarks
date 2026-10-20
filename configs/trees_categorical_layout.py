"""
Memory layout of the categorical-encoded X in trees and forests.

scikit-learn trees encode the categorical features of a DataFrame into a
float32 array before building the trees, and predict on the same layout.
This compares a C-ordered (main) and an F-ordered encoded array, see
scikit-learn PR #35162 and its split.

The data are pandas DataFrames with `category` columns (the "long-tail"
columns of `make_trees_*_data`), fitted with
`categorical_features="from_dtype"`: "long-tail" makes every column
categorical, "mix" about a third of them.
"""
from itertools import product

from joblib import cpu_count

from _utils.implementations import implementations_for_pixi_env

SHAPES = [
    {"n_samples": 200_000, "n_features": 20, "n_informative": 10},
    {"n_samples": 20_000, "n_features": 200, "n_informative": 50},
]
FORESTS = [
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
]
TREES = ["DecisionTreeClassifier", "DecisionTreeRegressor"]
BENCH = {"n_runs": 5, "time_limit": 600, "py_spy_profiling": False}


def _case(implem, estimator, shape, columns):
    generation_kwargs = {**shape, "columns": columns, "as_frame": True}
    if estimator.endswith("Classifier"):
        source = "make_trees_classification_data"
        generation_kwargs.update({"n_classes": 2, "n_redundant": 0})
    else:
        source = "make_trees_regression_data"
        generation_kwargs["noise"] = 0.1

    params = {"categorical_features": "from_dtype"}
    if estimator in FORESTS:
        params.update(
            {"n_estimators": cpu_count(), "max_features": 0.3, "n_jobs": -1}
        )
    return {
        "bench": BENCH,
        "algorithm": {"estimator": estimator, "estimator_params": params},
        "data": {"source": source, "generation_kwargs": generation_kwargs},
        "implementation": implem,
    }


def generate_cases() -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        for estimator, shape, columns in product(
            FORESTS, SHAPES, ["mix", "long-tail"]
        ):
            cases.append(_case(implem, estimator, shape, columns))
        for estimator, shape in product(TREES, SHAPES):
            cases.append(_case(implem, estimator, shape, "mix"))
    return cases
