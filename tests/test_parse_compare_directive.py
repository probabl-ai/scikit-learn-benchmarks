"""Tests for the `sklbench-compare` directive parser.

`scripts/parse_compare_directive.py` is a workflow script, not a package
module, so load it by path. The GPU case is covered here because the
directive is what decides whether the AWS L4 host gets provisioned at all.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "parse_compare_directive.py"


def _load_parser():
    spec = importlib.util.spec_from_file_location("parse_compare_directive", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = _load_parser()


def _directive(runs: str) -> str:
    return f"sklearn_ref: some-user:feature\nruns: {runs}\n"


def test_gpu_runner_with_source_built_cuda_env_is_accepted():
    parsed = parser.parse_fields(
        _directive("aws-nvidia-l4#sklearn-dev-nvidia#configs/all_models.py")
    )
    assert parsed["runs"] == [
        {
            "runner": "aws-nvidia-l4",
            "env": "sklearn-dev-nvidia",
            "config": "configs/all_models.py",
        }
    ]
    assert parsed["runners"] == ["aws-nvidia-l4"]


def test_unknown_runner_is_rejected_with_accepted_values():
    with pytest.raises(parser.Malformed) as excinfo:
        parser.parse_fields(_directive("intel-nope#configs/all_models.py"))

    message = str(excinfo.value)
    assert "intel-nope" in message
    for runner in ("intel-laptop", "intel-xeon", "aws-nvidia-l4"):
        assert runner in message


def test_non_source_built_env_is_rejected_with_accepted_values():
    with pytest.raises(parser.Malformed) as excinfo:
        parser.parse_fields(_directive("aws-nvidia-l4#sklearn-pypi#configs/all_models.py"))

    message = str(excinfo.value)
    assert "sklearn-pypi" in message
    for env in (
        "sklearn-dev",
        "sklearn-dev-libomp",
        "sklearn-dev-mkl",
        "sklearn-dev-freethreading",
        "sklearn-dev-nvidia",
    ):
        assert env in message


def test_default_runner_stays_intel_only():
    parsed = parser.parse_fields(_directive("configs/all_models.py"))
    assert parsed["runners"] == ["intel-laptop", "intel-xeon"]
    assert parsed["runs"] == [
        {"runner": "intel-laptop", "env": "sklearn-dev", "config": "configs/all_models.py"},
        {"runner": "intel-xeon", "env": "sklearn-dev", "config": "configs/all_models.py"},
    ]


def test_intel_only_directive_is_unchanged():
    parsed = parser.parse_fields(
        _directive(
            "intel-xeon#sklearn-dev-libomp#configs/pipeline.py, "
            "intel-laptop#sklearn-dev-mkl#configs/all_models.py"
        )
    )
    assert parsed["runners"] == ["intel-xeon", "intel-laptop"]
    assert parsed["runs"] == [
        {"runner": "intel-xeon", "env": "sklearn-dev-libomp", "config": "configs/pipeline.py"},
        {"runner": "intel-laptop", "env": "sklearn-dev-mkl", "config": "configs/all_models.py"},
    ]


def test_cli_writes_gpu_run_as_json(tmp_path):
    body = tmp_path / "body.md"
    body.write_text(
        "Intro text.\n\n```sklbench-compare\n"
        "sklearn_ref: some-user:feature\n"
        "runs: aws-nvidia-l4#sklearn-dev-nvidia#configs/all_models.py\n"
        "```\n"
    )
    github_output = tmp_path / "github_output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--body-file",
            str(body),
            "--github-output",
            str(github_output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    outputs = dict(
        line.split("=", 1)
        for line in github_output.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("runs<<")
    )
    assert outputs["ok"] == "true"
    assert json.loads(outputs["runs"]) == [
        {
            "runner": "aws-nvidia-l4",
            "env": "sklearn-dev-nvidia",
            "config": "configs/all_models.py",
        }
    ]
