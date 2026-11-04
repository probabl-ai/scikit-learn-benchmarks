"""Properties the benchmark configs must keep, whatever cases they define."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from dashboards.gen_hardware_comparisons import comparison_key, search_match_key
from sklbench.config import load_cases_from_script
from sklbench.reporting.matching import _case_without_bench


CONFIGS_DIR = Path("configs").resolve()


@pytest.fixture(autouse=True)
def every_gpu_is_available(monkeypatch):
    for name in ("_oneapi_gpu_available", "_nvidia_gpu_available", "_mps_gpu_available"):
        monkeypatch.setattr(f"sklbench.config.utils.{name}", lambda: True)


def _forget_config_modules():
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().is_relative_to(CONFIGS_DIR):
            del sys.modules[name]


@contextmanager
def _machine(n_cpus):
    """Configs read the core count at import time."""
    _forget_config_modules()
    with (
        mock.patch("os.cpu_count", lambda: n_cpus),
        mock.patch("os.sched_getaffinity", lambda pid: set(range(n_cpus)), create=True),
        mock.patch("psutil.cpu_count", lambda logical=True: n_cpus if logical else n_cpus // 2),
    ):
        try:
            yield
        finally:
            _forget_config_modules()


def _cases(monkeypatch, config, pixi_env):
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", pixi_env)
    return load_cases_from_script(config)


def _identities(cases, key):
    return {
        key(SimpleNamespace(case=_case_without_bench(case.json_dict()))) for case in cases
    }


@pytest.mark.parametrize(
    "config, key",
    [
        ("configs/all_models.py", comparison_key),
        ("configs/all_models_16gb.py", comparison_key),
        ("configs/hptuning.py", search_match_key),
    ],
)
@pytest.mark.parametrize("pixi_env", ["sklearn-pypi", "intel"])
def test_every_machine_runs_the_same_cases(monkeypatch, config, key, pixi_env):
    """The hardware and cross comparisons can only compare what pairs up, so a
    case that depends on the machine silently vanishes from them."""
    with _machine(8):
        small = _identities(_cases(monkeypatch, config, pixi_env), key)
    with _machine(344):
        big = _identities(_cases(monkeypatch, config, pixi_env), key)

    assert small == big
