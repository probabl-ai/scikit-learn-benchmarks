import subprocess
import sys
from pathlib import Path

from sklbench.orchestrator import env


def test_parse_openmp_display_env():
    output = """
OPENMP DISPLAY ENVIRONMENT BEGIN
  _OPENMP = '201511'
  OMP_NUM_THREADS = '16'
  OMP_WAIT_POLICY = 'PASSIVE'
  GOMP_SPINCOUNT = '300000'
OPENMP DISPLAY ENVIRONMENT END
"""

    assert env._parse_openmp_display_env(output) == {
        "_OPENMP": "201511",
        "OMP_NUM_THREADS": "16",
        "OMP_WAIT_POLICY": "PASSIVE",
        "GOMP_SPINCOUNT": "300000",
    }


def test_get_openmp_runtime_info(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="""
OPENMP DISPLAY ENVIRONMENT BEGIN
  OMP_NUM_THREADS = '2'
  GOMP_SPINCOUNT = '300000'
OPENMP DISPLAY ENVIRONMENT END
""",
        )

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    assert env.get_openmp_runtime_info() == {
        "OMP_NUM_THREADS": "2",
        "GOMP_SPINCOUNT": "300000",
    }
    command, kwargs = calls[0]
    assert command == [
        sys.executable,
        "-c",
        "from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled",
    ]
    assert kwargs["env"]["OMP_DISPLAY_ENV"] == "VERBOSE"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False
    assert kwargs["text"] is True


def test_git_info_for_path_ignores_benchmark_repo_root(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "scikit-learn-benchmarks"
    module_file = (
        benchmark_root
        / ".pixi/envs/sklearn/lib/python3.12/site-packages/sklearn/__init__.py"
    )

    monkeypatch.setattr(env, "_benchmark_repo_git_root", lambda: benchmark_root)

    def fake_check_output(command, cwd=None):
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return str(benchmark_root)
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(env, "_check_output", fake_check_output)

    assert env._git_info_for_path(module_file) is None


def test_git_info_for_path_keeps_nested_checkout(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "scikit-learn-benchmarks"
    sklearn_root = benchmark_root / ".bench/sklearn-worktrees/pr"
    module_file = sklearn_root / "sklearn/__init__.py"

    monkeypatch.setattr(env, "_benchmark_repo_git_root", lambda: benchmark_root)

    def fake_check_output(command, cwd=None):
        cwd = Path(cwd)
        if command == ["git", "rev-parse", "--show-toplevel"]:
            assert cwd == module_file.parent
            return str(sklearn_root)
        if command == ["git", "rev-parse", "HEAD"]:
            assert cwd == sklearn_root
            return "abc123"
        if command == ["git", "branch", "--show-current"]:
            assert cwd == sklearn_root
            return "feature"
        if command == ["git", "status", "--porcelain", "--untracked-files=no"]:
            assert cwd == sklearn_root
            return ""
        if command == ["git", "describe", "--tags", "--always", "--dirty"]:
            assert cwd == sklearn_root
            return "abc123"
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(env, "_check_output", fake_check_output)

    assert env._git_info_for_path(module_file) == {
        "commit": "abc123",
        "dirty": False,
        "branch": "feature",
        "describe": "abc123",
    }
