import os

import pytest

from pathlib import Path

from sklbench.config import Algorithm, Bench, Data, HPTuningCase
from sklbench.orchestrator import commands
from sklbench.orchestrator.commands import run_runner_from_case, runner_env


def _hptuning_case(**kwargs):
    return HPTuningCase(
        algorithm=Algorithm(estimator="Ridge"),
        data=Data(source="make_regression"),
        **kwargs,
    )


def test_runner_env_merges_bench_env_on_top_of_ambient_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "344")

    env = runner_env(_hptuning_case(bench=Bench(env={"OMP_NUM_THREADS": "128"})))

    assert env["OMP_NUM_THREADS"] == "128"
    assert env["PATH"] == os.environ["PATH"]


def _fake_run_runner_once(calls):
    def fake(bench_case, n_runs, bench_time_limit, py_spy_output, cprofile_output):
        calls.append(
            {
                "n_runs": n_runs,
                "bench_time_limit": bench_time_limit,
                "py_spy_output": py_spy_output,
                "cprofile_output": cprofile_output,
            }
        )
        return 0, [{"repeat": len(calls)}], None

    return fake


def test_run_runner_from_case_defaults_to_one_subprocess_for_all_repeats(monkeypatch):
    calls = []
    monkeypatch.setattr(commands, "_run_runner_once", _fake_run_runner_once(calls))

    return_code, rows, failed_case = run_runner_from_case(
        _hptuning_case(bench=Bench(n_runs=3, subprocess_per_repeat=False))
    )

    assert return_code == 0
    assert failed_case is None
    assert [c["n_runs"] for c in calls] == [3]
    assert rows == [{"repeat": 1}]


def test_run_runner_from_case_isolates_each_repeat_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(commands, "_run_runner_once", _fake_run_runner_once(calls))

    return_code, rows, failed_case = run_runner_from_case(
        _hptuning_case(bench=Bench(n_runs=3, subprocess_per_repeat=True))
    )

    assert return_code == 0
    assert failed_case is None
    assert [c["n_runs"] for c in calls] == [1, 1, 1]
    assert rows == [{"repeat": 1}, {"repeat": 2}, {"repeat": 3}]


def test_isolated_repeats_stop_on_first_failure(monkeypatch):
    calls = []

    def fake(bench_case, n_runs, bench_time_limit, py_spy_output, cprofile_output):
        calls.append(n_runs)
        if len(calls) == 2:
            return -9, [], {"error": "timeout"}
        return 0, [{"repeat": len(calls)}], None

    monkeypatch.setattr(commands, "_run_runner_once", fake)

    return_code, rows, failed_case = run_runner_from_case(
        _hptuning_case(bench=Bench(n_runs=5, subprocess_per_repeat=True))
    )

    assert return_code == -9
    assert failed_case == {"error": "timeout"}
    # Stops after the failing repeat instead of running the remaining ones.
    assert calls == [1, 1]
    assert rows == [{"repeat": 1}]


def test_isolated_repeats_share_one_time_budget(monkeypatch):
    now = [0.0]
    budgets = []

    def fake(bench_case, n_runs, bench_time_limit, py_spy_output, cprofile_output):
        budgets.append(bench_time_limit)
        now[0] += 6.0
        return 0, [{"repeat": len(budgets)}], None

    monkeypatch.setattr(commands.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(commands, "_run_runner_once", fake)

    return_code, rows, _ = run_runner_from_case(
        _hptuning_case(bench=Bench(n_runs=5, time_limit=10, subprocess_per_repeat=True))
    )

    assert return_code == 0
    assert len(rows) == 2
    assert budgets == pytest.approx([10, 4])


def test_isolated_repeats_bypass_for_py_spy_and_cprofile_passes(monkeypatch):
    calls = []
    monkeypatch.setattr(commands, "_run_runner_once", _fake_run_runner_once(calls))

    run_runner_from_case(
        _hptuning_case(bench=Bench(n_runs=3, subprocess_per_repeat=True)),
        py_spy_output=Path("/tmp/profile.raw"),
        n_runs_override=1,
    )

    assert [c["n_runs"] for c in calls] == [1]
    assert calls[0]["py_spy_output"] == Path("/tmp/profile.raw")
