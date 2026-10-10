import os

import pytest

from pathlib import Path

from sklbench.config import Algorithm, Bench, Data, HPTuningCase
from sklbench.orchestrator import commands
from sklbench.orchestrator.commands import (
    pin_process_affinity,
    run_runner_from_case,
    runner_env,
)


def _hptuning_case(**kwargs):
    return HPTuningCase(
        algorithm=Algorithm(estimator="Ridge"),
        data=Data(source="make_regression"),
        **kwargs,
    )


def test_runner_env_defaults_to_ambient_environment():
    env = runner_env(_hptuning_case(bench=Bench()))

    assert env == os.environ


def test_runner_env_merges_bench_env_on_top_of_ambient_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "344")

    env = runner_env(_hptuning_case(bench=Bench(env={"OMP_NUM_THREADS": "128"})))

    assert env["OMP_NUM_THREADS"] == "128"
    assert env["PATH"] == os.environ["PATH"]


def test_pin_process_affinity_sets_the_given_core_list(monkeypatch):
    calls = []

    class FakeProcess:
        def __init__(self, pid):
            calls.append(pid)

        def cpu_affinity(self, cores):
            calls.append(cores)

    monkeypatch.setattr(commands.psutil, "Process", FakeProcess)

    pin_process_affinity(1234, [0, 1, 3])

    assert calls == [1234, [0, 1, 3]]


def test_pin_process_affinity_raises_where_unsupported(monkeypatch):
    # On macOS, psutil.Process doesn't define cpu_affinity at all (it's only
    # added to the class on Linux/Windows/FreeBSD) - accessing it raises
    # AttributeError. pin_process_affinity must not swallow this: configs
    # setting bench.cpu_affinity are responsible for not doing so on
    # unsupported platforms (see configs/smoke_check_test.py), not the other
    # way around - confirmed the hard way on the macOS CI runner
    # (macos-setup-check.yml), which crashed until that guard was added.
    class FakeProcess:
        def __init__(self, pid):
            pass

    monkeypatch.setattr(commands.psutil, "Process", FakeProcess)

    with pytest.raises(AttributeError):
        pin_process_affinity(1234, [0, 1])


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


def test_isolated_repeats_stops_once_the_shared_time_budget_is_exhausted(monkeypatch):
    calls = []
    remaining_budgets = []

    def fake(bench_case, n_runs, bench_time_limit, py_spy_output, cprofile_output):
        calls.append(n_runs)
        remaining_budgets.append(bench_time_limit)
        return 0, [{"repeat": len(calls)}], None

    times = iter([0.0, 0.0, 5.0, 11.0])
    monkeypatch.setattr(commands.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(commands, "_run_runner_once", fake)

    return_code, rows, failed_case = run_runner_from_case(
        _hptuning_case(bench=Bench(n_runs=5, time_limit=10, subprocess_per_repeat=True))
    )

    assert return_code == 0
    # Budget (10s from t=0) is spent after 2 repeats (checked at t=5, t=11):
    # only the first two get to run.
    assert calls == [1, 1]
    assert len(rows) == 2


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
