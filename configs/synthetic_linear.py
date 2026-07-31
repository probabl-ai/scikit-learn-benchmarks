from itertools import chain, product
from math import ceil
from typing import Iterable

#TODO: in utils or in _common? wierd overlap
from _common import deterministic_random_choice


ALGORITHM_VARIANTS = [
    {"estimator": "Ridge"},
    {"estimator": "RidgeClassifier"},
    {"estimator": "LinearRegression"},  # TODO: remove?
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "lbfgs"}},
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "newton-cholesky"}},
]


def _build_case(
    bench: dict, algorithm: dict, generation_kwargs: dict, implem: dict
) -> dict:
    is_classifier = algorithm["estimator"] in {"LogisticRegression", "RidgeClassifier"}
    source = "make_classification" if is_classifier else "make_regression"
    # TODO: include multi class a bit
    extra_generation_kwargs = {"n_classes": 2, "n_redundant": 0} if is_classifier else {}
    data = {
        "source": source,
        "generation_kwargs": {**generation_kwargs, **extra_generation_kwargs},
    }
    # TODO: really check both order for everything?
    for order in ['C', 'F']:
        return {
            "bench": bench,
            "algorithm": algorithm,
            "data": {**data, 'order': order},
            "implementation": implem,
        }


def _linear_cases_for(implem: dict, bench: dict, data_variants: list[dict]) -> Iterable[dict]:
    for algorithm, generation_kwargs in product(
        ALGORITHM_VARIANTS, data_variants
    ):
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

    yield from _linear_cases_for(implem, bench, data_variants)


def _scaled_linear_cases(implem: dict, scale: int) -> Iterable[dict]:
    """`_linear_data_shapes` scaled up/down - scale=10 matches the "fast" tier,
    so "normal"'s first rung sits at roughly the same magnitude as "fast",
    then grows from there - mirroring `synthetic_trees.py`'s scale ladder.
    """
    bench = {"time_limit": scale}

    data_shapes = [
        {"n_samples": 500000 * scale, "n_features": 2, "n_informative": 2},
        {"n_samples": 50000 * scale, "n_features": 20, "n_informative": 5},
        {"n_samples": 5000 * scale, "n_features": 200, "n_informative": 40},
        {"n_samples": 500 * scale, "n_features": 2000, "n_informative": 100},
    ]

    yield from _linear_cases_for(implem, bench, data_shapes)


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
        if deterministic_random_choice(case, [0, 0, 1])
    ]

    return cases
