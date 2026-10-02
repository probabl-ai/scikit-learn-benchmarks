from all_models import generate_cases as generate_all_models_cases
from _utils.implementations import implementations_for_pixi_env
from _synthetic_linear import _linear_cases_for, linear_data_shapes
from _utils.numa import auto_socket_cpu_affinity
from sklbench.config import Bench


ALGORITHM = {"estimator": "LogisticRegression", "estimator_params": {"solver": "lbfgs"}}

# Extra synthetic-data scales beyond all_models.py's own "normal" tier ladder
# (20, 80) - the lbfgs order-preservation PR (scikit-learn#34903) only pays
# off once the F-order GEMV in LinearModelLoss.loss_gradient is big enough
# for BLAS threading to matter, so a wider scale sweep is more informative
# here than for the general matrix.
EXTRA_SCALES = [10, 40, 60]

# The PR only changes anything by way of BLAS thread parallelization (see
# scikit-learn#34903's description) - sweep low/high thread counts via env
# var so the comparison actually exercises that axis instead of whatever
# thread count happens to be the ambient default on the runner.
BLAS_THREAD_COUNTS = [1, 4, None]

# The PR's effect is specific to F-order X (see LinearModelLoss.loss_gradient's
# GEMV). Force both orders explicitly on every case - including real datasets,
# which real_datasets.py otherwise loads as C-order - so the sweep covers the
# PR's affected path (F) and its unaffected control (C).
DATA_ORDERS = ["C", "F"]

N_RUNS = 10

# Pin to one socket on multi-socket runners (intel-gnr), so "all threads"
# means one socket's cores and memory rather than BLAS threads spread over
# both sockets and their cross-socket memory traffic. None (no-op) on
# single-socket runners.
CPU_AFFINITY = auto_socket_cpu_affinity(socket=0)


def _is_target(case) -> bool:
    solver = case.algorithm.estimator_params.get("solver", "lbfgs")
    if case.algorithm.estimator != "LogisticRegression" or solver != "lbfgs":
        return False
    # covtype's LogisticRegression case is too slow for this sweep's already
    # wide (scale x thread-count x order) matrix.
    if case.data.dataset == "covtype":
        return False
    return True


def _extra_scale_cases() -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        for scale in EXTRA_SCALES:
            data_shapes = linear_data_shapes(scale)
            benchs = [{"time_limit": 2 + scale * 2} for _ in data_shapes]
            cases.extend(_linear_cases_for(implem, ALGORITHM, benchs, data_shapes))
    return cases


def _with_blas_threads(case: dict, n_threads: int | None) -> dict:
    bench = case.get("bench") or {}
    # time_limit is a total budget over all repeats: scale it with the
    # repeat count so the extra repeats aren't cut off by it.
    n_runs = bench.get("n_runs") or Bench().n_runs
    time_limit = (bench.get("time_limit") or Bench().time_limit) * N_RUNS / n_runs
    env = {}
    if n_threads is not None:
        env["OPENBLAS_NUM_THREADS"] = str(n_threads)
    return {
        **case,
        "metadata": {**case.get("metadata", {}), "blas_num_threads": n_threads},
        "bench": {
            **bench,
            "n_runs": N_RUNS,
            "time_limit": time_limit,
            "cpu_affinity": CPU_AFFINITY,
            "py_spy_profiling": False,
            # This sweep's before/after comparison is exactly the shape
            # issue #80 warns about: a single-process draw of NUMA memory
            # placement can look like a solver regression. Isolate each
            # repeat into its own subprocess so the reported number is an
            # average over draws instead of one of them.
            "subprocess_per_repeat": True,
            "env": {
                **(bench.get("env") or {}),
                **env
            },
        },
    }


def _with_order(case: dict, order: str) -> dict:
    return {**case, "data": {**case.get("data", {}), "order": order}}


def generate_cases() -> list[dict]:
    # all_models.generate_cases() returns EstimatorCase objects, not plain
    # dicts, despite its own type hint - its last filtering step
    # (filter_array_api_supported_cases_if_needed) converts every case via
    # EstimatorCase(**case) before yielding it.
    cases = [
        case.model_dump(mode="json")
        for case in generate_all_models_cases()
        if _is_target(case)
    ]
    cases.extend(_extra_scale_cases())

    return [
        _with_blas_threads(_with_order(case, order), n_threads)
        for case in cases
        for order in DATA_ORDERS
        for n_threads in BLAS_THREAD_COUNTS
    ]
