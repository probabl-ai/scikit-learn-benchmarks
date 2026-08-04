from copy import deepcopy
from itertools import chain, product
from math import ceil
from typing import Iterable

#TODO: in utils or in _common? wierd overlap
from _common import deterministic_random_choice


def _stable_seed(case: dict) -> dict:
    """Strip `case["implementation"]` before hashing so the one-third
    subsample in `generate_cases` picks the same cases regardless of which
    implementation (sklearn/sklearnex/array-API) generated them. Without
    this, each implementation independently samples a different subset of
    the matrix, breaking the sklearn-vs-sklearnex/array-API comparison for
    most cases (see the same fix in synthetic_trees.py's `_stable_seed`).
    """
    seed = deepcopy(case)
    seed.pop("implementation", None)
    return seed


ALGORITHM_VARIANTS = [
    {"estimator": "Ridge"},
    # {"estimator": "RidgeClassifier"},
    # ^ not really interesting on synthetic dataset for benchmarks (just Ridge under the hoods)
    {"estimator": "LinearRegression"},  # TODO: remove?
    # TODO: in dashboards: take the best & fastest of the 3 solvers:
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "lbfgs"}},
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "newton-cholesky"}},
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "newton-cg"}},
]


def _build_case(
    bench: dict, algorithm: dict, generation_kwargs: dict, implem: dict
) -> dict:
    is_classifier = algorithm["estimator"] in {"LogisticRegression", "RidgeClassifier"}
    source = "make_classification" if is_classifier else "make_regression"

    n_features = generation_kwargs["n_features"]
    extra_generation_kwargs = {}
    if is_classifier and n_features >= 10:
        seed = [generation_kwargs, algorithm['estimator'], source]
        n_classes = deterministic_random_choice(seed, [2, 2, 5])
        n_redundant = deterministic_random_choice(seed, [0, 5, n_features // 2])
        extra_generation_kwargs.update(
            {"n_classes": n_classes, "n_redundant": n_redundant}
        )
    elif is_classifier:
        extra_generation_kwargs.update({"n_classes": 2, "n_redundant": 0})

    case = {
        "bench": bench,
        "algorithm": algorithm,
        "data": {
            "source": source,
            "generation_kwargs": {
                **generation_kwargs, **extra_generation_kwargs},
        },
    }
    # TODO: only F/only C? only 32/64?
    case['data']['order'] = deterministic_random_choice(case, ['C', 'F'])
    case['data']['dtype'] = deterministic_random_choice(case, ['float32', 'float64'])
    case["implementation"] = implem

    return case


def _linear_cases_for(implem: dict, benchs: list[dict], data_variants: list[dict]) -> Iterable[dict]:
    for algorithm, (bench, generation_kwargs) in product(
        ALGORITHM_VARIANTS, zip(benchs, data_variants)
    ):
        solver = algorithm.get("estimator_params", {}).get("solver")
        if solver in ("newton-cholesky", "newton-cg") and generation_kwargs["n_features"] >= 1000:
            # Newton solvers factorize the (n_features, n_features) Hessian each
            # iteration, so they don't scale to wide feature spaces - times out
            # regardless of implementation rather than measuring anything useful.
            continue
        yield _build_case(bench, algorithm, generation_kwargs, implem)


def _test_linear_cases(implem: dict) -> Iterable[dict]:
    bench = {"n_runs": 3}
    data_variants = [
        {
            "n_samples": 1000,
            "n_features": 20,
            "n_informative": ceil(0.5 * 20),
        }
    ]

    yield from _linear_cases_for(implem, [bench], data_variants)


def linear_data_shapes(scale: int) -> list[dict]:
    """Base (n_samples, n_features, n_informative) shapes at a given scale.

    Shared with `all_models_scaling.py`'s thread-count scaling study so both
    configs stay in sync on what "scale N" means. scale=10 matches the
    "fast" tier, so "normal"'s first rung sits at roughly the same magnitude
    as "fast", then grows from there - mirroring `synthetic_trees.py`'s
    scale ladder.
    """
    return [
        {"n_samples": 500000 * scale, "n_features": 2, "n_informative": 2},
        {"n_samples": 50000 * scale, "n_features": 20, "n_informative": 5},
        {"n_samples": 5000 * scale, "n_features": 200, "n_informative": 40},
        {"n_samples": 500 * scale, "n_features": 2000, "n_informative": 100},
    ]


def _scaled_linear_cases(implem: dict, scale: int) -> Iterable[dict]:
    data_shapes = linear_data_shapes(scale)

    benchs = [{"time_limit": 2 + scale * 2} for _ in data_shapes]
    if implem['library'] == 'sklearn':
        # 2 features is very slow in sklearn
        benchs[0]["time_limit"] *= 2

    yield from _linear_cases_for(implem, benchs, data_shapes)


def generate_cases(implem: dict | None = None, tier: str = "normal") -> list[dict]:
    if implem is None:
        implem = {"library": "sklearn"}

    if tier == "fast":
        return list(_scaled_linear_cases(implem, scale=10))
    if tier == "test":
        return list(_test_linear_cases(implem))

    cases = list(chain(
        _scaled_linear_cases(implem, scale=10),
        _scaled_linear_cases(implem, scale=20),
        _scaled_linear_cases(implem, scale=50),
        _scaled_linear_cases(implem, scale=100),
    ))

    if tier == "slow":
        cases += list(chain(
            _scaled_linear_cases(implem, scale=200),
            _scaled_linear_cases(implem, scale=500),
            _scaled_linear_cases(implem, scale=1000),
        ))

    # sub-sample one third of the matrix
    cases = [
        case for case in cases
        if deterministic_random_choice(_stable_seed(case), [0, 0, 1])
    ]

    return cases
