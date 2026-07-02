from pathlib import Path

from sklbench.config import EstimatorCase, PipelineCase, load_cases_from_script


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


def test_shipped_configs_generate_valid_cases():
    config_paths = sorted(
        path for path in Path("configs").glob("*.py") if not path.name.startswith("_")
    )

    cases_by_config = {
        path.name: load_cases_from_script(path)
        for path in config_paths
    }

    assert set(cases_by_config) == {
        "array_api_cpu.py",
        "array_api_intel.py",
        "array_api_nvidia.py",
        "hgb_scaling.py",
        "pipeline.py",
        "sklearn.py",
        "sklearnex_cpu.py",
        "sklearnex_gpu.py",
    }
    assert all(cases for cases in cases_by_config.values())
