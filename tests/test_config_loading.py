import json
from pathlib import Path

import pytest

from sklbench.config import BaseCase, EstimatorCase, load_cases_from_script


SKLEARN_ENVS = [
    "sklearn-pypi",
    "sklearn-cf-default",
    "sklearn-cf-libgomp-openblas",
    "sklearn-cf-libomp-openblas",
    "sklearn-cf-libomp-openblas-omp",
    "sklearn-cf-mkl",
    "sklearn-dev",
]
GENERAL_ENVS = [*SKLEARN_ENVS, "intel"]
ARRAY_API_ENVS = [*GENERAL_ENVS, "skl-cpu", "skl-intel", "skl-nvidia", "skl-mps"]

ENV_SENSITIVE_CONFIGS = {
    Path("configs/smoke_check_test.py"): [*ARRAY_API_ENVS, "cuml"],
    Path("configs/all_models.py"): [*ARRAY_API_ENVS, "cuml"],
    Path("configs/all_models_16gb.py"): ARRAY_API_ENVS,
    Path("configs/all_models_cuml_only.py"): [*SKLEARN_ENVS, "cuml"],
    Path("configs/models_scalability.py"): GENERAL_ENVS,
    # hptuning.py only ever selects non-array-API (no data_library),
    # CPU-or-unset-device implementations - the array-API-only envs
    # (skl-cpu/skl-intel/skl-nvidia/skl-mps) would filter down to zero
    # implementations and produce no cases.
    Path("configs/hptuning.py"): GENERAL_ENVS,
    Path("configs/hptuning_taskset.py"): GENERAL_ENVS,
}


def test_loaded_cases_record_their_config(tmp_path):
    """Dashboards pick their results by this stamp."""
    config = tmp_path / "config.py"
    config.write_text(
        """
def generate_cases():
    return [{
        "implementation": {"library": "sklearn"},
        "algorithm": {"estimator": "Ridge"},
        "data": {"source": "make_regression"},
    }]
""",
        encoding="utf-8",
    )

    (case,) = load_cases_from_script(config)

    assert case.metadata["source_config"].endswith("config.py")


def test_shipped_configs_generate_valid_cases(monkeypatch):
    # This checks that each config *can* produce cases for every Pixi
    # environment it supports, including GPU-only ones (skl-intel,
    # skl-nvidia) - independent of whether the machine actually running the
    # test has that GPU hardware. `filter_gpu_cases_if_unavailable` (see
    # test_filter_gpu_cases_drops_gpu_cases_without_matching_hardware and
    # test_filter_gpu_cases_keeps_gpu_cases_with_matching_hardware below) is
    # what makes that hardware check environment-dependent for real runs;
    # pretend a GPU is always present here so this test keeps validating
    # config *structure*.
    monkeypatch.setattr("sklbench.config.utils._oneapi_gpu_available", lambda: True)
    monkeypatch.setattr("sklbench.config.utils._nvidia_gpu_available", lambda: True)
    monkeypatch.setattr("sklbench.config.utils._mps_gpu_available", lambda: True)

    config_paths = sorted(
        path for path in Path("configs").glob("*.py") if not path.name.startswith("_")
    )

    for path in config_paths:
        if path in ENV_SENSITIVE_CONFIGS:
            continue
        monkeypatch.delenv("PIXI_ENVIRONMENT_NAME", raising=False)
        load_cases_from_script(path)

    for path, pixi_envs in ENV_SENSITIVE_CONFIGS.items():
        for pixi_env in pixi_envs:
            monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", pixi_env)

            cases = load_cases_from_script(path)

            assert cases
            assert all(isinstance(case, BaseCase) for case in cases)


def test_env_sensitive_configs_require_pixi_environment(monkeypatch):
    monkeypatch.delenv("PIXI_ENVIRONMENT_NAME", raising=False)

    with pytest.raises(ValueError, match="PIXI_ENVIRONMENT_NAME is not set"):
        load_cases_from_script("configs/smoke_check_test.py")


def test_env_sensitive_configs_reject_unknown_pixi_environment(monkeypatch):
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "unknown")

    with pytest.raises(ValueError, match="Unsupported PIXI_ENVIRONMENT_NAME"):
        load_cases_from_script("configs/smoke_check_test.py")


def test_all_models_configs_support_array_api_pixi_environments(monkeypatch):
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "skl-cpu")

    cases = load_cases_from_script("configs/smoke_check_test.py")

    assert cases
    assert all(isinstance(case, EstimatorCase) for case in cases)
    # hgb_scalability.py's cases (metadata.benchmark_type == "scaling") are
    # always plain sklearn regardless of Pixi environment - its
    # thread-scaling/cpu_affinity sweep isn't an array-API concept - so
    # they're excluded from this assertion.
    non_scaling_cases = [
        case for case in cases if case.metadata.get("benchmark_type") != "scaling"
    ]
    assert non_scaling_cases
    assert all(case.implementation.is_array_api() for case in non_scaling_cases)


def _gpu_and_cpu_cases():
    return [
        {
            "algorithm": {"estimator": "LogisticRegression"},
            "data": {"source": "make_classification"},
            "implementation": {"library": "sklearnex", "device": "gpu"},
        },
        {
            "algorithm": {"estimator": "Ridge"},
            "data": {"source": "make_regression"},
            "implementation": {"library": "sklearn", "device": "cuda"},
        },
        {
            "algorithm": {"estimator": "Ridge"},
            "data": {"source": "make_regression"},
            "implementation": {"library": "sklearn", "device": "mps"},
        },
        {
            "algorithm": {"estimator": "Ridge"},
            "data": {"source": "make_regression"},
            "implementation": {"library": "sklearnex", "device": "cpu"},
        },
        {
            "algorithm": {"estimator": "Ridge"},
            "data": {"source": "make_regression"},
            "implementation": {"library": "sklearn"},
        },
    ]


def test_filter_gpu_cases_drops_gpu_cases_without_matching_hardware(monkeypatch):
    from sklbench.config.utils import filter_gpu_cases_if_unavailable

    monkeypatch.setattr("sklbench.config.utils._oneapi_gpu_available", lambda: False)
    monkeypatch.setattr("sklbench.config.utils._nvidia_gpu_available", lambda: False)
    monkeypatch.setattr("sklbench.config.utils._mps_gpu_available", lambda: False)

    kept = list(filter_gpu_cases_if_unavailable(_gpu_and_cpu_cases()))

    assert [case["implementation"]["device"] for case in kept if "device" in case["implementation"]] == ["cpu"]
    assert len(kept) == 2


def test_filter_gpu_cases_keeps_gpu_cases_with_matching_hardware(monkeypatch):
    from sklbench.config.utils import filter_gpu_cases_if_unavailable

    monkeypatch.setattr("sklbench.config.utils._oneapi_gpu_available", lambda: True)
    monkeypatch.setattr("sklbench.config.utils._nvidia_gpu_available", lambda: False)
    monkeypatch.setattr("sklbench.config.utils._mps_gpu_available", lambda: True)

    kept = list(filter_gpu_cases_if_unavailable(_gpu_and_cpu_cases()))

    assert [case["implementation"].get("device") for case in kept] == ["gpu", "mps", "cpu", None]


def _cuml_filter_cases():
    cuml = {"library": "cuml", "device": "cuda"}
    sklearn = {"library": "sklearn"}
    return [
        {
            "algorithm": {"estimator": "LinearRegression"},
            "data": {"source": "make_regression"},
            "implementation": cuml,
        },
        {
            "algorithm": {
                "estimator": "LogisticRegression",
                "estimator_params": {"solver": "newton-cg"},
            },
            "data": {"source": "make_classification"},
            "implementation": cuml,
        },
        {
            "algorithm": {
                "estimator": "LogisticRegression",
                "estimator_params": {"solver": "lbfgs"},
            },
            "data": {"source": "make_classification"},
            "implementation": cuml,
        },
        {
            "algorithm": {
                "estimator": "LogisticRegression",
                "estimator_params": {"solver": "newton-cholesky", "max_iter": 1000},
            },
            "data": {"dataset": "kddcup09_churn"},
            "implementation": cuml,
        },
        {
            "algorithm": {"estimator": "LinearRegression"},
            "data": {"source": "make_regression"},
            "implementation": sklearn,
        },
    ]


def test_filter_cuml_supported_cases_drops_only_cuml_cases():
    from sklbench.config.utils import filter_cuml_supported_cases_if_needed

    kept = list(filter_cuml_supported_cases_if_needed(_cuml_filter_cases()))

    assert [(case["algorithm"]["estimator"], case["implementation"]["library"]) for case in kept] == [
        ("LogisticRegression", "cuml"),
        ("LogisticRegression", "cuml"),
        ("LinearRegression", "sklearn"),
    ]
    assert kept[0]["algorithm"]["estimator_params"]["solver"] == "lbfgs"
    assert kept[1]["data"]["dataset"] == "kddcup09_churn"


def _case_identity(case: EstimatorCase) -> str:
    payload = case.model_dump(mode="json")
    payload.pop("implementation")
    return json.dumps(payload, sort_keys=True)


def test_all_models_filters_cuml_and_keeps_the_sklearn_suite(monkeypatch):
    monkeypatch.setattr("sklbench.config.utils._nvidia_gpu_available", lambda: True)

    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "sklearn-pypi")
    sklearn_cases = load_cases_from_script("configs/all_models.py")
    sklearn_estimators = {case.algorithm.estimator for case in sklearn_cases}
    assert "LinearRegression" in sklearn_estimators
    assert "ExtraTreesClassifier" in sklearn_estimators

    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "cuml")
    cuml_cases = load_cases_from_script("configs/all_models.py")
    assert {case.algorithm.estimator for case in cuml_cases} == {
        "LogisticRegression",
        "Ridge",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "KMeans",
    }
    synthetic_lr_solvers = {
        case.algorithm.estimator_params.get("solver")
        for case in cuml_cases
        if case.algorithm.estimator == "LogisticRegression" and case.data.dataset is None
    }
    assert synthetic_lr_solvers == {"lbfgs"}
    real_lr_solvers = {
        case.algorithm.estimator_params.get("solver")
        for case in cuml_cases
        if case.algorithm.estimator == "LogisticRegression" and case.data.dataset is not None
    }
    assert "newton-cholesky" in real_lr_solvers


def test_all_models_cuml_only_matches_sklearn_and_cuml(monkeypatch):
    monkeypatch.setattr("sklbench.config.utils._nvidia_gpu_available", lambda: True)

    identities = {}
    for pixi_env in ("sklearn-pypi", "cuml"):
        monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", pixi_env)
        cases = load_cases_from_script("configs/all_models_cuml_only.py")
        assert {case.algorithm.estimator for case in cases} == {
            "LogisticRegression",
            "Ridge",
            "RandomForestClassifier",
            "RandomForestRegressor",
            "KMeans",
        }
        identities[pixi_env] = {_case_identity(case) for case in cases}

    assert identities["sklearn-pypi"] == identities["cuml"]
