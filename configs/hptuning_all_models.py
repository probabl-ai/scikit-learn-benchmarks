"""
Realistic hyper-parameter searches: one search per (model, dataset) pair, run
with `n_jobs=-1` on the whole machine.

Each search fits `n_candidates` seeded random points of its `space` on
`n_splits` shuffled folds. The fits are single-threaded (forests keep their
default `n_jobs`, and joblib limits OpenMP and BLAS to one thread per worker),
so the search's own `n_jobs=-1` is the only level of parallelism.

Searches have 640 to 2,560 fits, enough to feed every logical CPU of a large
server, and take from ~10 minutes to ~2.3 hours of single-core time, ~11.5 hours
in total. The estimate next to each `n_splits` is extrapolated from the first
8 candidates, on a Core Ultra X7 358H.

Linear models mostly tune the preprocessing, which changes the shape of the
problem given to the solver (left at its default). Tree models mostly tune
their own hyper-parameters, plus the encoding of the categorical columns.

A `space` maps a pipeline parameter to its list of values. An entry whose
values are dicts sets several parameters at once (for example an encoding and
the options that only apply to it).
"""

import numpy as np

from _utils.implementations import implementations_for_pixi_env

from sklbench.config import Algorithm, Data, HPTuning, HPTuningCase

# One run of the whole config already takes hours on a machine with few cores.
BENCH = {"n_runs": 2, "time_limit": 8 * 3600}

SPLINES = "preprocessor__numeric__"
ONE_HOT = "preprocessor__categorical__onehot__"
# With `preprocessing_kwargs={"nystroem": {}}`, the column transformer is one
# step of the preprocessor.
NYSTROEM_SPLINES = "preprocessor__columntransformer__numeric__"

# The linear preprocessing appends the target encoding of the categorical
# columns to their one-hot encoding.
WITH_OR_WITHOUT_TARGET_ENCODING = [{}, {"preprocessor__categorical__target": "drop"}]


def _tree_encodings(one_hot_max_categories):
    encoder = "preprocessor__encoder__"
    return [
        {f"{encoder}encoding": "ordinal", f"{encoder}min_frequency": 5},
        {f"{encoder}encoding": "ordinal", f"{encoder}min_frequency": 50},
        {f"{encoder}encoding": "target"},
        *(
            {f"{encoder}encoding": "one-hot", f"{encoder}max_categories": max_categories}
            for max_categories in one_hot_max_categories
        ),
    ]


def _logspace(start, stop, num):
    return [float(f"{value:.3g}") for value in np.logspace(start, stop, num)]


SEARCHES = [
    # LogisticRegression (lbfgs, the default solver)
    dict(
        estimator="LogisticRegression",
        dataset="bank_marketing",
        preprocessing_kind="linear",
        n_candidates=512,
        n_splits=3,  # ~15 min
        estimator_params={"max_iter": 1000},
        space={
            f"{SPLINES}n_knots": [3, 5, 8, 12, 20, 30],
            f"{SPLINES}degree": [1, 2, 3],
            f"{ONE_HOT}max_categories": [4, 8, 16],
            "target_encoding": WITH_OR_WITHOUT_TARGET_ENCODING,
            "estimator__C": _logspace(-3, 3, 13),
        },
    ),
    dict(
        # Categorical columns only, with up to 7,500 categories each.
        estimator="LogisticRegression",
        dataset="amazon_employee_access",
        preprocessing_kind="linear",
        n_candidates=256,
        n_splits=5,  # ~20 min
        estimator_params={"max_iter": 1000},
        space={
            f"{ONE_HOT}max_categories": [4, 8, 16, 32, 64, 128],
            f"{ONE_HOT}min_frequency": [1, 5, 20],
            "target_encoding": WITH_OR_WITHOUT_TARGET_ENCODING,
            "estimator__C": _logspace(-3, 3, 13),
        },
    ),
    dict(
        # 26 classes, numeric features only.
        estimator="LogisticRegression",
        dataset="letters",
        preprocessing_kind="linear",
        n_candidates=128,
        n_splits=5,  # ~45 min
        estimator_params={"max_iter": 1000},
        space={
            f"{SPLINES}n_knots": [3, 5, 8, 12, 20],
            f"{SPLINES}degree": [1, 2, 3],
            "estimator__C": _logspace(-2, 4, 13),
        },
    ),
    # Ridge (solver="auto")
    dict(
        estimator="Ridge",
        dataset="california_housing",
        preprocessing_kind="linear",
        preprocessing_kwargs={"nystroem": {}},
        n_candidates=512,
        n_splits=5,  # ~30 min
        space={
            f"{NYSTROEM_SPLINES}n_knots": [5, 10, 20, 40],
            f"{NYSTROEM_SPLINES}degree": [1, 2, 3],
            "preprocessor__nystroem__n_components": [100, 200, 400, 800],
            "preprocessor__nystroem__degree": [2, 3],
            "estimator__alpha": _logspace(-4, 2, 13),
        },
    ),
    dict(
        # Categorical columns with up to 3,300 categories.
        estimator="Ridge",
        dataset="medical_charges_nominal",
        preprocessing_kind="linear",
        n_candidates=256,
        n_splits=5,  # ~20 min
        space={
            f"{SPLINES}n_knots": [5, 10, 20],
            f"{SPLINES}degree": [1, 2, 3],
            f"{ONE_HOT}max_categories": [4, 8, 16, 32],
            "target_encoding": WITH_OR_WITHOUT_TARGET_ENCODING,
            "estimator__alpha": _logspace(-3, 3, 7),
        },
    ),
    dict(
        # 81 numeric features: 243 to 2,106 after the splines.
        estimator="Ridge",
        dataset="superconduct",
        preprocessing_kind="linear",
        n_candidates=256,
        n_splits=5,  # ~10 min
        space={
            f"{SPLINES}n_knots": [3, 4, 6, 8, 12, 16, 24],
            f"{SPLINES}degree": [1, 2, 3],
            "estimator__alpha": _logspace(-3, 3, 13),
        },
    ),
    # Random forests
    dict(
        estimator="RandomForestClassifier",
        dataset="kddcup09_churn",
        preprocessing_kind="trees",
        n_candidates=128,
        n_splits=5,  # ~2.3 h
        estimator_params={"n_estimators": 100},
        space={
            "encoding": _tree_encodings(one_hot_max_categories=[4, 10]),
            "estimator__max_features": ["sqrt", 0.1, 0.3],
            "estimator__min_samples_leaf": [1, 5, 20, 50],
            "estimator__max_depth": [8, 16, None],
            "estimator__max_samples": [0.5, None],
        },
    ),
    dict(
        estimator="RandomForestRegressor",
        dataset="diamonds",
        preprocessing_kind="trees",
        n_candidates=128,
        n_splits=5,  # ~1.3 h
        estimator_params={"n_estimators": 100},
        space={
            "encoding": _tree_encodings(one_hot_max_categories=[4, 8]),
            "estimator__max_features": [1.0, 0.5, 0.3],
            "estimator__min_samples_leaf": [1, 5, 20],
            "estimator__max_depth": [8, 16, None],
            "estimator__max_samples": [0.5, None],
        },
    ),
    dict(
        # Categorical columns only.
        estimator="RandomForestClassifier",
        dataset="amazon_employee_access",
        preprocessing_kind="trees",
        n_candidates=128,
        n_splits=5,  # ~20 min
        estimator_params={"n_estimators": 100},
        space={
            "encoding": _tree_encodings(one_hot_max_categories=[8, 32]),
            "estimator__max_features": ["sqrt", 0.5, 1.0],
            "estimator__min_samples_leaf": [1, 2, 5, 20],
            "estimator__max_depth": [10, 20, None],
            "estimator__max_samples": [0.5, None],
        },
    ),
    # Extra-trees
    dict(
        estimator="ExtraTreesRegressor",
        dataset="california_housing",
        n_candidates=512,
        n_splits=5,  # ~50 min
        estimator_params={"n_estimators": 200},
        space={
            "estimator__max_features": [1.0, 0.75, 0.5, 0.25],
            "estimator__min_samples_leaf": [1, 2, 5, 10, 20],
            "estimator__min_samples_split": [2, 10, 30],
            "estimator__max_depth": [8, 12, 16, 24, None],
            "estimator__bootstrap": [False, True],
        },
    ),
    dict(
        # Small data (1,460 rows), 43 categorical columns.
        estimator="ExtraTreesRegressor",
        dataset="ames_housing",
        preprocessing_kind="trees",
        n_candidates=512,
        n_splits=5,  # ~2 h
        estimator_params={"n_estimators": 500},
        space={
            "encoding": _tree_encodings(one_hot_max_categories=[4, 10]),
            "estimator__max_features": [1.0, 0.5, 0.3, 0.1],
            "estimator__min_samples_leaf": [1, 2, 5, 10],
            "estimator__max_depth": [6, 10, 16, None],
            "estimator__bootstrap": [False, True],
        },
    ),
    # HistGradientBoosting. No early stopping, so that the number of
    # iterations doesn't depend on the machine.
    dict(
        # Native categorical splits on 18 columns.
        estimator="HistGradientBoostingClassifier",
        dataset="kick",
        preprocessing_kind="hgb",
        n_candidates=128,
        n_splits=5,  # ~40 min
        estimator_params={"early_stopping": False},
        space={
            "preprocessor__max_categories": [16, 64, 252],
            "preprocessor__min_frequency": [5, 20],
            "estimator__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "estimator__max_iter": [100, 200, 400],
            "estimator__max_leaf_nodes": [7, 15, 31, 63],
            "estimator__min_samples_leaf": [5, 20, 50],
            "estimator__l2_regularization": [0.0, 0.1, 1.0],
        },
    ),
    dict(
        # Native categorical splits, on columns with up to 3,300 categories.
        estimator="HistGradientBoostingRegressor",
        dataset="medical_charges_nominal",
        preprocessing_kind="hgb",
        n_candidates=128,
        n_splits=5,  # ~30 min
        estimator_params={"early_stopping": False},
        space={
            "preprocessor__max_categories": [16, 64, 252],
            "preprocessor__min_frequency": [5, 20],
            "estimator__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "estimator__max_iter": [100, 200, 400],
            "estimator__max_leaf_nodes": [15, 31, 63, 127],
            "estimator__min_samples_leaf": [5, 20, 50, 200],
            "estimator__l2_regularization": [0.0, 0.1, 1.0],
            "estimator__max_features": [0.5, 1.0],
        },
    ),
    dict(
        # 26 classes: 26 trees per iteration.
        estimator="HistGradientBoostingClassifier",
        dataset="letters",
        preprocessing_kind="hgb",
        n_candidates=128,
        n_splits=5,  # ~1.2 h
        estimator_params={"early_stopping": False},
        space={
            "estimator__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "estimator__max_iter": [50, 100, 200],
            "estimator__max_leaf_nodes": [7, 15, 31],
            "estimator__min_samples_leaf": [5, 20, 50],
            "estimator__l2_regularization": [0.0, 0.1, 1.0],
            "estimator__max_features": [0.5, 1.0],
        },
    ),
]


def _candidates(space: dict, n_candidates: int, seed: int = 0) -> list[dict]:
    """`n_candidates` distinct points of `space`, the same on every machine
    (legacy `RandomState` streams don't change across numpy versions)."""
    names = sorted(space)
    shape = [len(space[name]) for name in names]
    n_points = int(np.prod(shape))
    if n_points < n_candidates:
        raise ValueError(f"The space has {n_points} points, fewer than {n_candidates}")
    indices = np.random.RandomState(seed).choice(n_points, size=n_candidates, replace=False)
    candidates = []
    # Kept in drawing order: sorted, the fits running at the same time would
    # share most of their parameters.
    for point in zip(*np.unravel_index(indices, shape)):
        candidate = {}
        for name, index in zip(names, point):
            value = space[name][index]
            candidate.update(value if isinstance(value, dict) else {name: value})
        candidates.append(candidate)
    return candidates


def generate_cases() -> list[HPTuningCase]:
    # scikit-learn on CPU only for now.
    implementations = [
        implem
        for implem in implementations_for_pixi_env()
        if implem["library"] == "sklearn"
        and implem.get("data_library") is None
        and implem.get("device") in (None, "cpu")
    ]

    cases = []
    for search in SEARCHES:
        for implem in implementations:
            cases.append(HPTuningCase(
                bench=BENCH,
                algorithm=Algorithm(
                    estimator=search["estimator"],
                    estimator_params=search.get("estimator_params", {}),
                ),
                data=Data(
                    dataset=search["dataset"],
                    preprocessing_kind=search.get("preprocessing_kind"),
                    preprocessing_kwargs=search.get("preprocessing_kwargs", {}),
                ),
                implementation=implem,
                hptuning=HPTuning(
                    candidates=_candidates(search["space"], search["n_candidates"]),
                    cv="kfold",
                    n_splits=search["n_splits"],
                    n_jobs=-1,
                    random_state=3198,
                ),
            ))
    return cases
