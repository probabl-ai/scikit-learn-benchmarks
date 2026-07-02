import logging
import subprocess as sp
import sys

from sklbench.config import Bench, PipelineCase
from sklbench.orchestrator import implementation


def test_call_benchmarks_logs_subprocess_failure(tmp_path, monkeypatch, caplog):
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
    caplog.set_level(logging.WARNING, logger=implementation.logger.name)

    return_code, _, failed_cases = implementation.call_benchmarks(
        [case],
        hardware_hash="hardware",
        software_hash="software",
        results_dir=str(tmp_path),
    )

    assert return_code == 2
    assert failed_cases == [failed_case]
    assert "Benchmark failed for" in caplog.text
    assert "return code 2" in caplog.text
    assert "runner exploded" in caplog.text
    assert "partial output" in caplog.text


def test_call_benchmarks_logs_setup_failure(tmp_path, monkeypatch, caplog):
    case = PipelineCase(bench=Bench(py_spy_profiling=False))

    def fake_run_runner_from_case(bench_case):
        raise sp.SubprocessError("unable to start runner")

    monkeypatch.setattr(
        implementation, "run_runner_from_case", fake_run_runner_from_case
    )
    caplog.set_level(logging.WARNING, logger=implementation.logger.name)

    return_code, _, failed_cases = implementation.call_benchmarks(
        [case],
        hardware_hash="hardware",
        software_hash="software",
        results_dir=str(tmp_path),
    )

    assert return_code == -1
    assert len(failed_cases) == 1
    assert "Benchmark setup failed for" in caplog.text
    assert "return code -1" in caplog.text
    assert "unable to start runner" in caplog.text
