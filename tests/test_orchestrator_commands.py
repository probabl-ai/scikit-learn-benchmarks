import os

import pytest

from sklbench.config import Algorithm, Bench, Data, HPTuningCase
from sklbench.orchestrator import commands
from sklbench.orchestrator.commands import pin_process_affinity, runner_env


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
    # unsupported platforms (see configs/all_models_test.py), not the other
    # way around - confirmed the hard way on the macOS CI runner
    # (macos-setup-check.yml), which crashed until that guard was added.
    class FakeProcess:
        def __init__(self, pid):
            pass

    monkeypatch.setattr(commands.psutil, "Process", FakeProcess)

    with pytest.raises(AttributeError):
        pin_process_affinity(1234, [0, 1])
