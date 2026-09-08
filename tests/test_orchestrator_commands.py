import os
import sys
from pathlib import Path

from sklbench.config import Algorithm, Bench, Data, HPTuningCase
from sklbench.orchestrator.commands import generate_runner_command, runner_env


def _hptuning_case(**kwargs):
    return HPTuningCase(
        algorithm=Algorithm(estimator="Ridge"),
        data=Data(source="make_regression"),
        **kwargs,
    )


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


def test_runner_env_defaults_to_ambient_environment():
    env = runner_env(_hptuning_case(bench=Bench()))

    assert env == os.environ


def test_runner_env_merges_bench_env_on_top_of_ambient_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "344")

    env = runner_env(_hptuning_case(bench=Bench(env={"OMP_NUM_THREADS": "128"})))

    assert env["OMP_NUM_THREADS"] == "128"
    assert env["PATH"] == os.environ["PATH"]
