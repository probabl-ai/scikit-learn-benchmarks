"""Real benchmark runs, from the orchestrator to the dashboards. The other dashboard
tests write records by hand, so only these notice a change of the record format."""

from argparse import Namespace
import json
from pathlib import Path
import statistics
import subprocess as sp

import pytest

from sklbench.config import EstimatorCase
from sklbench.orchestrator import implementation
from sklbench.reporting.matching import read_all_results, read_failed_records
from test_speedup_comparison import ENVS, LAPTOP, Results, on_machine, only, points, table


REPO_ROOT = Path(__file__).resolve().parent.parent


def _case(**estimator_params):
    return EstimatorCase(
        algorithm={"estimator": "Ridge", "estimator_params": estimator_params},
        data={"source": "make_regression", "generation_kwargs": {"n_samples": 400, "n_features": 5}},
        implementation={"library": "sklearn"},
        metadata={"source_config": "configs/all_models.py"},
        bench={"n_runs": 3, "time_limit": 120, "py_spy_profiling": False},
    )


def _orchestrate(monkeypatch, cases, software_hash):
    """Everything is real except the environment description, which would make the
    results depend on the machine running the tests."""
    env_info = {
        "hardware": {"CPU": {"name": "test CPU"}},
        "software": {"pixi_environment_name": ENVS[software_hash], "pixi_packages": {"python": {"version": "3.14"}}},
    }
    monkeypatch.setattr(implementation, "get_environment_info", lambda: env_info)
    monkeypatch.setattr(implementation, "get_hardware_hash", lambda info: LAPTOP)
    monkeypatch.setattr(implementation, "get_software_hash", lambda info: software_hash)
    return implementation.orchestrate_benchmarks(
        cases,
        # "test" in the config name skips the warmup, like smoke runs.
        Namespace(
            results_dir="results",
            exit_on_error=False,
            config=["test"],
            system_telemetry_interval=2.0,
            system_telemetry_percpu=False,
        ),
    )


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    # The runners are subprocesses started from `tmp_path`.
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    monkeypatch.chdir(tmp_path)
    Results(tmp_path)
    return tmp_path / "results"


def _raw_times(results_dir, software_hash, method):
    for path in (results_dir / "records").glob("*.json"):
        record = json.loads(path.read_text())
        if record["software_hash"] == software_hash and record["results"]:
            return [run["time_ms"][method] for run in record["results"]]
    raise AssertionError(f"no record for {software_hash}")


def test_real_runs_reach_the_dashboards_with_their_own_times(results_dir, monkeypatch):
    assert _orchestrate(monkeypatch, [_case()], "pypi00") == 0
    assert _orchestrate(monkeypatch, [_case()], "mkl000") == 0

    assert {(result.method, len(result.times)) for result in read_all_results()} == {
        ("preprocessing", 3),
        ("fit", 3),
        ("predict", 3),
    }
    expected = statistics.median(_raw_times(results_dir, "pypi00", "fit")) / statistics.median(
        _raw_times(results_dir, "mkl000", "fit")
    )
    rendered = on_machine(LAPTOP, softwareGroups=["non-pypi"])
    assert only(points(rendered, "linear")).speedup == pytest.approx(expected, rel=1e-2)

    rows = {row["variant"]: row for row in table(rendered, "linear")}
    candidate = rows["sklearn-cf-mkl"]
    assert candidate["n_samples"] > 0 and candidate["n_features"] == 5
    assert candidate["status"] == "ok"
    assert 0 <= candidate["r2"] <= 1


def test_each_repeat_of_a_real_run_records_its_seeds(results_dir, monkeypatch):
    assert _orchestrate(monkeypatch, [_case()], "pypi00") == 0

    (path,) = (results_dir / "records").glob("*.json")
    runs = json.loads(path.read_text())["results"]
    assert [run["random_state"]["data"] for run in runs] == [0, 1, 2]


def test_a_failing_run_is_recorded_and_doesnt_stop_the_next_ones(results_dir, monkeypatch):
    return_code = _orchestrate(monkeypatch, [_case(no_such_param=1), _case()], "pypi00")

    assert return_code != 0
    (failed,) = read_failed_records()
    assert failed.failed_case["return_code"] == return_code
    assert "no_such_param" in failed.failed_case["logs"]["stderr"]
    assert {result.method for result in read_all_results()} >= {"fit", "predict"}


def test_a_runner_that_cant_start_is_recorded_as_failed(results_dir, monkeypatch):
    def cant_start(bench_case):
        raise sp.SubprocessError("unable to start runner")

    monkeypatch.setattr(implementation, "run_runner_from_case", cant_start)

    assert _orchestrate(monkeypatch, [_case()], "pypi00") == -1
    (failed,) = read_failed_records()
    assert failed.failed_case["return_code"] == -1
