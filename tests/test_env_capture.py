import json
import subprocess

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


def test_get_openmp_runtime_info_reads_the_real_runtime():
    assert "_OPENMP" in env.get_openmp_runtime_info()


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=someone", "-c", "user.email=s@example.com", *args],
        check=True,
        capture_output=True,
    )


def _repo(path, tracked_file):
    tracked = path / tracked_file
    tracked.parent.mkdir(parents=True)
    tracked.write_text("v1")
    _git(path, "init", "-b", "feature")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "first")
    return tracked


def test_git_info_ignores_a_package_installed_inside_the_benchmark_repo(tmp_path, monkeypatch):
    benchmarks = tmp_path / "benchmarks"
    module = _repo(benchmarks, ".pixi/envs/sklearn/lib/sklearn/__init__.py")
    monkeypatch.setattr(env, "_benchmark_repo_git_root", lambda: benchmarks.resolve())

    assert env._git_info_for_path(module) is None


def test_git_info_describes_a_nested_scikit_learn_checkout(tmp_path, monkeypatch):
    benchmarks = tmp_path / "benchmarks"
    _repo(benchmarks, "README.md")
    module = _repo(benchmarks / "sklearn-src", "sklearn/__init__.py")
    _git(module.parents[1], "remote", "add", "origin", "git@github.com:scikit-learn/scikit-learn.git")
    monkeypatch.setattr(env, "_benchmark_repo_git_root", lambda: benchmarks.resolve())

    info = env._git_info_for_path(module)
    assert info["branch"] == "feature"
    assert info["remote"] == "git@github.com:scikit-learn/scikit-learn.git"
    assert len(info["commit"]) == 40
    assert info["dirty"] is False

    module.write_text("v2")
    assert env._git_info_for_path(module)["dirty"] is True


def test_get_oneapi_devices_via_intel_env_falls_back_when_pixi_project_root_unset(
    monkeypatch,
):
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)
    assert env._get_oneapi_devices_via_intel_env() == {}


def test_get_oneapi_devices_via_intel_env_falls_back_when_the_subprocess_fails(
    monkeypatch,
):
    monkeypatch.setenv("PIXI_PROJECT_ROOT", "/repo")
    monkeypatch.setattr(env, "_check_output", lambda command: None)
    assert env._get_oneapi_devices_via_intel_env() == {}


def test_get_oneapi_devices_via_intel_env_reads_the_intel_env_report(monkeypatch):
    monkeypatch.setenv("PIXI_PROJECT_ROOT", "/repo")
    device = {"level_zero:gpu:0": {"name": "Intel(R) Arc(TM) B390 GPU"}}
    monkeypatch.setattr(env, "_check_output", lambda command: json.dumps(device))

    assert env._get_oneapi_devices_via_intel_env() == device
