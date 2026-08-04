from pathlib import Path

import pytest

from sklbench.config import EstimatorCase, PipelineCase, load_cases_from_script


SKLEARN_ENVS = [
    "sklearn",
    "sklearn-conda",
    "sklearn-openblas-pthreads",
    "sklearn-openblas-openmp",
    "sklearn-mkl",
    "sklearn-dev",
]
GENERAL_ENVS = [*SKLEARN_ENVS, "intel"]
ARRAY_API_ENVS = [*GENERAL_ENVS, "skl-cpu", "skl-intel", "skl-nvidia"]

ENV_SENSITIVE_CONFIGS = {
    Path("configs/all_models_test.py"): ARRAY_API_ENVS,
    Path("configs/all_models_fast.py"): ARRAY_API_ENVS,
    Path("configs/all_models.py"): ARRAY_API_ENVS,
    Path("configs/all_models_scaling.py"): GENERAL_ENVS,
}


def test_load_cases_from_script_accepts_model_instances(tmp_path):
    config = tmp_path / "config.py"
    config.write_text(
        """
from sklbench.config import PipelineCase


def generate_cases():
    return [PipelineCase()]
""",
        encoding="utf-8",
    )

    cases = load_cases_from_script(config)

    assert len(cases) == 1
    assert isinstance(cases[0], PipelineCase)


def test_load_cases_from_script_accepts_estimator_dict(tmp_path):
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

    cases = load_cases_from_script(config)

    assert len(cases) == 1
    assert isinstance(cases[0], EstimatorCase)
    assert cases[0].metadata == {}


def test_estimator_case_routes_to_estimator_runner():
    case = EstimatorCase(
        algorithm={"estimator": "Ridge"},
        data={"source": "make_regression"},
        implementation={"library": "sklearn"},
    )

    assert case.runner_module == "sklbench.runners.estimator"


def test_shipped_configs_generate_valid_cases(monkeypatch):
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
            assert all(isinstance(case, EstimatorCase) for case in cases)


def test_env_sensitive_configs_require_pixi_environment(monkeypatch):
    monkeypatch.delenv("PIXI_ENVIRONMENT_NAME", raising=False)

    with pytest.raises(ValueError, match="PIXI_ENVIRONMENT_NAME is not set"):
        load_cases_from_script("configs/all_models_test.py")


def test_env_sensitive_configs_reject_unknown_pixi_environment(monkeypatch):
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "unknown")

    with pytest.raises(ValueError, match="Unsupported PIXI_ENVIRONMENT_NAME"):
        load_cases_from_script("configs/all_models_test.py")


def test_all_models_configs_support_array_api_pixi_environments(monkeypatch):
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "skl-cpu")

    cases = load_cases_from_script("configs/all_models_test.py")

    assert cases
    assert all(isinstance(case, EstimatorCase) for case in cases)
    assert all(case.implementation.is_array_api() for case in cases)


def test_filter_array_api_supported_cases_excludes_sklearnex_ridge_classifier():
    from sklbench.config.utils import filter_array_api_supported_cases_if_needed

    cases = [
        {
            "algorithm": {
                "estimator": "RidgeClassifier",
                "estimator_params": {"solver": "svd"},
            },
            "data": {"source": "make_classification"},
            "implementation": {
                "library": "sklearnex",
                "sklearnex_context": {"array_api_dispatch": True},
            },
        },
        {
            "algorithm": {"estimator": "Ridge", "estimator_params": {"solver": "svd"}},
            "data": {"source": "make_regression"},
            "implementation": {
                "library": "sklearn",
                "sklearn_context": {"array_api_dispatch": True},
            },
        },
    ]

    kept = list(filter_array_api_supported_cases_if_needed(cases))

    assert [case.algorithm.estimator for case in kept] == ["Ridge"]
