import os
import sys
from pathlib import Path

from sklbench.config import Bench, PipelineCase
from sklbench.orchestrator.commands import generate_runner_command, runner_env


def test_generate_runner_command_enables_native_py_spy_profiling():
    command = generate_runner_command(
        PipelineCase(bench=Bench()),
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
        "sklbench.runners.pipeline",
        "--case-file",
        "case.json",
        "--n-runs",
        "3",
        "--output-jsonl",
        "results.jsonl",
    ]


def test_runner_env_defaults_to_ambient_environment():
    env = runner_env(PipelineCase(bench=Bench()))

    assert env == os.environ


def test_runner_env_merges_bench_env_on_top_of_ambient_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "344")

    env = runner_env(PipelineCase(bench=Bench(env={"OMP_NUM_THREADS": "128"})))

    assert env["OMP_NUM_THREADS"] == "128"
    assert env["PATH"] == os.environ["PATH"]
