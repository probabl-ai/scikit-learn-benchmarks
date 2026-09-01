from itertools import chain
from math import ceil, sqrt
from typing import Iterable

#TODO: in utils or in _common? wierd overlap
from _common import deterministic_random_choice


ALGORITHM_VARIANTS = [
    {"estimator": "Ridge"},
    {"estimator": "LinearRegression"},
    # TODO: in dashboards: take the best & fastest of the 3 solvers:
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "lbfgs"}},
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "newton-cholesky"}},
    {"estimator": "LogisticRegression", "estimator_params": {"solver": "newton-cg"}},
]

# Per-estimator multiplier applied on top of the base scale in scaled tiers,
# so each estimator's data size grows at its own rate.
ALGORITHM_SCALE_FACTORS = {
    "LinearRegression": 1,
    "LogisticRegression": 2,
    "Ridge": 5,
}


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
    # order="F" is known to be extremely slow (sometimes timing out) for
    # sklearnex GPU Ridge/LinearRegression/LogisticRegression - see
    # https://github.com/uxlfoundation/scikit-learn-intelex/issues/3235.
    # Already fixed upstream in oneDAL (uxlfoundation/oneDAL#3665, merged
    # 2026-07-10) but not yet in a scikit-learn-intelex release (latest is
    # 2026.1.0, from 2026-06-10) - leaving order in the mix as-is for now,
    # remove this comment once a release with the fix is picked up.
    case['data']['order'] = deterministic_random_choice(case, ['C', 'F'])
    case['data']['dtype'] = deterministic_random_choice(case, ['float32', 'float64'])
    case["implementation"] = implem

    return case


def _linear_cases_for(
    implem: dict, algorithm: dict, benchs: list[dict], data_variants: list[dict]
) -> Iterable[dict]:
    for bench, generation_kwargs in zip(benchs, data_variants):
        solver = algorithm.get("estimator_params", {}).get("solver")
        if solver in ("newton-cholesky", "newton-cg") and generation_kwargs["n_features"] >= 1000:
            # Newton solvers factorize the (n_features, n_features) Hessian each
            # iteration, so they don't scale to wide feature spaces - times out
            # regardless of implementation rather than measuring anything useful.
            continue

        case = _build_case(bench, algorithm, generation_kwargs, implem)
        if (
            implem.get("device") == "xpu"
            and case["data"]["dtype"] == "float64"
            and generation_kwargs["n_samples"] > 10_000_000
        ):
            # XPU (torch) is broken for float64 once n_samples goes past ~10M -
            # crashes rather than measuring anything useful.
            # https://github.com/intel/torch-xpu-ops/issues/4805
            continue
        yield case


def _test_linear_cases(implem: dict) -> Iterable[dict]:
    bench = {"n_runs": 3}
    data_variants = [
        {
            "n_samples": 1000,
            "n_features": 20,
            "n_informative": ceil(0.5 * 20),
        }
    ]

    for algorithm in ALGORITHM_VARIANTS:
        yield from _linear_cases_for(implem, algorithm, [bench], data_variants)


def linear_data_shapes(scale: int) -> list[dict]:
    """Base (n_samples, n_features, n_informative) shapes at a given scale.

    Shared with `all_models_scaling.py`'s thread-count scaling study so both
    configs stay in sync on what "scale N" means. scale=10 matches the
    "fast" tier, so "normal"'s first rung sits at roughly the same magnitude
    as "fast", then grows from there - mirroring `synthetic_trees.py`'s
    scale ladder.
    """
    return [
        {"n_samples": 50000 * scale, "n_features": 20, "n_informative": 5},
        {"n_samples": 5000 * scale, "n_features": 100, "n_informative": 20},
        {"n_samples": 500 * scale, "n_features": 1000, "n_informative": 100},
        {
            "n_samples": int(round(250 * sqrt(scale), -2)),
            "n_features": int(round(2000 * sqrt(scale), -2)),
            "n_informative": 500
        },
    ]


def _scaled_linear_cases(implem: dict, scale: int) -> Iterable[dict]:
    cases = []
    for algorithm in ALGORITHM_VARIANTS:
        algorithm_scale = scale * ALGORITHM_SCALE_FACTORS[algorithm["estimator"]]
        data_shapes = linear_data_shapes(algorithm_scale)
        benchs = [{"time_limit": 2 + algorithm_scale * 2} for _ in data_shapes]
        cases.extend(_linear_cases_for(implem, algorithm, benchs, data_shapes))

    return cases


def generate_cases(implem: dict | None = None, tier: str = "normal") -> list[dict]:
    if implem is None:
        implem = {"library": "sklearn"}

    if tier == "test":
        return list(_test_linear_cases(implem))

    scales = {
        "fast": [10],
        "normal": [20, 80],
    }

    cases = list(chain(*[
        _scaled_linear_cases(implem, scale=scale)
        for scale in scales[tier]
    ]))

    return cases
