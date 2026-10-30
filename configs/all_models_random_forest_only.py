from all_models import generate_cases as generate_all_models_cases

# Only RandomForest: ExtraTrees uses the random splitter, which doesn't sort the
# feature values.
TREE_ESTIMATORS = ["RandomForestClassifier", "RandomForestRegressor"]


def generate_cases() -> list[dict]:
    # all_models.generate_cases() returns EstimatorCase objects, not plain
    # dicts (see all_models_linear_only.py).
    cases = [
        case
        for case in generate_all_models_cases()
        if case.algorithm.estimator in TREE_ESTIMATORS
    ]
    # Same cases, single-threaded with a few trees: measures the per-tree fit
    # time without the parallel speed-up.
    cases += [
        case.model_copy(
            deep=True,
            update={
                "algorithm": case.algorithm.model_copy(
                    update={
                        "estimator_params": {
                            **case.algorithm.estimator_params,
                            "n_estimators": 4,
                            "n_jobs": 1,
                        }
                    }
                )
            },
        )
        for case in cases
    ]
    for case in cases:
        case.bench.py_spy_profiling = False
    return cases
