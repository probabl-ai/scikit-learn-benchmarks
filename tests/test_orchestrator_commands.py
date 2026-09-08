import os
import sys
from pathlib import Path

import pytest

from sklbench.config import Algorithm, Bench, Data, HPTuningCase
from sklbench.orchestrator import commands
from sklbench.orchestrator.commands import generate_runner_command, runner_env


def _hptuning_case(**kwargs):
    return HPTuningCase(
        algorithm=Algorithm(estimator="Ridge"),
        data=Data(source="make_regression"),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _single_numa_node(monkeypatch):
    """Most tests here don't care about NUMA interleaving; make the host's
    actual topology irrelevant to them by pretending it's single-node,
    which is enough to make `numa_interleave_prefix` a no-op. Tests that do
    care re-patch `_numa_node_count` (or `numa_interleave_prefix` directly)
    themselves.
    """
    monkeypatch.setattr(commands, "_numa_node_count", lambda: 1)


def test_generate_runner_command_enables_native_py_spy_profiling():
    command = generate_runner_command(
        _hptuning_case(bench=Bench()),
        case_file=Path("case.json"),
        n_runs=3,
        output_jsonl=Path("results.jsonl"),
        py_spy_output=Path("profile.raw"),
    )

    assert command == [
        "py-spy",
        "record",
        "--native",
        "--rate",
        "100",
        "--format",
        "raw",
        "-o",
        "profile.raw",
        "--",
        sys.executable,
        "-m",
        "sklbench.runners.hptuning",
        "--case-file",
        "case.json",
        "--n-runs",
        "3",
        "--output-jsonl",
        "results.jsonl",
    ]


def test_generate_runner_command_adds_numa_interleave_prefix(monkeypatch):
    monkeypatch.setattr(
        commands, "numa_interleave_prefix", lambda: ("numactl", "--interleave=all")
    )

    command = generate_runner_command(
        _hptuning_case(bench=Bench()),
        case_file=Path("case.json"),
        n_runs=3,
        output_jsonl=Path("results.jsonl"),
    )

    assert command[:2] == ["numactl", "--interleave=all"]
    assert command[2:] == [
        sys.executable,
        "-m",
        "sklbench.runners.hptuning",
        "--case-file",
        "case.json",
        "--n-runs",
        "3",
        "--output-jsonl",
        "results.jsonl",
    ]


def test_generate_runner_command_skips_numa_interleave_with_taskset(monkeypatch):
    monkeypatch.setattr(
        commands, "numa_interleave_prefix", lambda: ("numactl", "--interleave=all")
    )

    command = generate_runner_command(
        _hptuning_case(bench=Bench(taskset="0-7")),
        case_file=Path("case.json"),
        n_runs=3,
        output_jsonl=Path("results.jsonl"),
    )

    assert command[:3] == ["taskset", "-c", "0-7"]
    assert "numactl" not in command


def test_numa_interleave_prefix_empty_with_a_single_node(monkeypatch):
    monkeypatch.setattr(commands, "_numa_node_count", lambda: 1)

    assert commands.numa_interleave_prefix() == ()


def test_numa_interleave_prefix_empty_when_numactl_is_not_installed(monkeypatch):
    monkeypatch.setattr(commands, "_numa_node_count", lambda: 2)
    monkeypatch.setattr(commands.shutil, "which", lambda _: None)

    assert commands.numa_interleave_prefix() == ()


def test_numa_interleave_prefix_present_with_multiple_nodes_and_numactl(monkeypatch):
    monkeypatch.setattr(commands, "_numa_node_count", lambda: 8)
    monkeypatch.setattr(commands.shutil, "which", lambda _: "/usr/bin/numactl")

    assert commands.numa_interleave_prefix() == ("numactl", "--interleave=all")


def test_runner_env_defaults_to_ambient_environment():
    env = runner_env(_hptuning_case(bench=Bench()))

    assert env == os.environ


def test_runner_env_merges_bench_env_on_top_of_ambient_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "344")

    env = runner_env(_hptuning_case(bench=Bench(env={"OMP_NUM_THREADS": "128"})))

    assert env["OMP_NUM_THREADS"] == "128"
    assert env["PATH"] == os.environ["PATH"]
