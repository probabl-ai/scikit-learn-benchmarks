from all_models import generate_cases as generate_all_models_cases
from _implementations import implementations_for_pixi_env
from synthetic_linear import _linear_cases_for, linear_data_shapes


ALGORITHM = {"estimator": "LogisticRegression", "estimator_params": {"solver": "lbfgs"}}

# Extra synthetic-data scales beyond all_models.py's own "normal" tier ladder
# (20, 80) - the lbfgs order-preservation PR (scikit-learn#34903) only pays
# off once the F-order GEMV in LinearModelLoss.loss_gradient is big enough
# for BLAS threading to matter, so a wider scale sweep is more informative
# here than for the general matrix.
EXTRA_SCALES = [10, 40, 60]


def _is_target(case) -> bool:
    solver = case.algorithm.estimator_params.get("solver", "lbfgs")
    return case.algorithm.estimator == "LogisticRegression" and solver == "lbfgs"


def _extra_scale_cases() -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        for scale in EXTRA_SCALES:
            data_shapes = linear_data_shapes(scale)
            benchs = [{"time_limit": 2 + scale * 2} for _ in data_shapes]
            cases.extend(_linear_cases_for(implem, ALGORITHM, benchs, data_shapes))
    return cases


def generate_cases() -> list[dict]:
    # all_models.generate_cases() returns EstimatorCase objects, not plain
    # dicts, despite its own type hint - its last filtering step
    # (filter_array_api_supported_cases_if_needed) converts every case via
    # EstimatorCase(**case) before yielding it.
    cases = [case for case in generate_all_models_cases() if _is_target(case)]
    for case in cases:
        case.bench.py_spy_profiling = False

    cases = [case.model_dump(mode="json") for case in cases]
    cases.extend(_extra_scale_cases())
    return cases
