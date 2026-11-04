"""What dashboards declare must exist, or `scripts/what_to_rerun.py` prints commands
that can't run."""

from pathlib import Path
import tomllib

import pytest

from dashboards import GPU_NAMES, HARDWARE_NAMES, HARDWARE_PLATFORMS
from dashboards.index import DASHBOARDS


@pytest.mark.xfail(
    strict=True,
    reason="configs/all_models_cuml_only.py is added by #142, which must remove this xfail",
)
def test_declared_configs_exist():
    missing = {
        config
        for _, module, _ in DASHBOARDS
        for config in module.SOURCE_CONFIGS
        if not Path(config).is_file()
    }
    assert not missing


def test_declared_envs_exist():
    with open("pixi.toml", "rb") as f:
        environments = tomllib.load(f)["environments"]
    missing = {
        env for _, module, _ in DASHBOARDS for env in module.SOURCE_ENVS if env not in environments
    }
    assert not missing


def test_every_gpu_and_platform_belongs_to_a_known_machine():
    assert set(GPU_NAMES) <= set(HARDWARE_NAMES)
    assert set(HARDWARE_PLATFORMS) == set(HARDWARE_NAMES)
