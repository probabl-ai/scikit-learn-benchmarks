import logging
import subprocess as sp
import sys
from argparse import Namespace

from sklbench.config import Bench, PipelineCase
from sklbench.orchestrator import implementation


def _args(tmp_path):
    return Namespace(results_dir=str(tmp_path), exit_on_error=False)


def test_orchestrate_benchmarks_logs_subprocess_failure(tmp_path, monkeypatch, caplog):
    case = PipelineCase(bench=Bench(py_spy_profiling=False))
    failed_case = {
        "case": case.json_dict(),
        "return_code": 2,
        "command": [sys.executable, "-m", "sklbench.runners.pipeline"],
        "logs": {"stdout": "partial output", "stderr": "runner exploded"},
    }

    def fake_run_runner_from_case(bench_case):
        return 2, [], failed_case

    monkeypatch.setattr(
        implementation, "run_runner_from_case", fake_run_runner_from_case
    )
    monkeypatch.setattr(
        implementation,
        "get_environment_info",
        lambda: {"hardware": {}, "software": {}},
    )
    caplog.set_level(logging.WARNING, logger=implementation.logger.name)

    return_code = implementation.orchestrate_benchmarks([case], _args(tmp_path))

    assert return_code == 2
    assert "Benchmark failed for" in caplog.text
    assert "return code 2" in caplog.text
    assert "runner exploded" in caplog.text
    assert "partial output" in caplog.text


def test_orchestrate_benchmarks_logs_setup_failure(tmp_path, monkeypatch, caplog):
    case = PipelineCase(bench=Bench(py_spy_profiling=False))

    def fake_run_runner_from_case(bench_case):
        raise sp.SubprocessError("unable to start runner")

    monkeypatch.setattr(
        implementation, "run_runner_from_case", fake_run_runner_from_case
    )
    monkeypatch.setattr(
        implementation,
        "get_environment_info",
        lambda: {"hardware": {}, "software": {}},
    )
    caplog.set_level(logging.WARNING, logger=implementation.logger.name)

    return_code = implementation.orchestrate_benchmarks([case], _args(tmp_path))

    assert return_code == -1
    assert "Benchmark setup failed for" in caplog.text
    assert "return code -1" in caplog.text
    assert "unable to start runner" in caplog.text
