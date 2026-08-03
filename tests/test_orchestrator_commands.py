import sys
from pathlib import Path

from sklbench.config import Bench, PipelineCase
from sklbench.orchestrator.commands import generate_runner_command


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
