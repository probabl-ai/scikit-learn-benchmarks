import json
import os
import subprocess as sp
import sys
import tempfile
from pathlib import Path

from ..config import Case, EstimatorCase


RUNNER_MODULES = {
    "estimator": "sklbench.runners.estimator",
    "pipeline": "sklbench.runners.pipeline",
}
PY_SPY_NO_CHILD_PROCESS_ERROR = "Error: No child process (os error 10)"


def _n_jobs(bench_case: Case) -> int:
    """Effective thread/process parallelism the benchmarked run will use.

    Falls back to `os.cpu_count()` for `n_jobs <= 0` (joblib/sklearn's
    "use all cores" convention), so the py-spy rate policy below reacts to
    actual parallelism rather than the literal sentinel value.
    """
    if isinstance(bench_case, EstimatorCase):
        n_jobs = bench_case.algorithm.estimator_params.get("n_jobs", 1)
    else:
        n_jobs = bench_case.run.n_jobs
    if not n_jobs or n_jobs <= 0:
        return os.cpu_count() or 1
    return n_jobs


def py_spy_rate(bench_case: Case) -> int:
    """py-spy sampling rate (Hz), lowered as `n_jobs` grows.

    High-`n_jobs` cases push more threads through py-spy's ptrace-based
    per-tick stack walk, which can trigger a scheduler-churn feedback loop
    that makes profiling fall pathologically behind real time (see
    https://github.com/cakedev0/scikit-learn/issues/14). Lowering the rate
    keeps profiling overhead bounded on those cases.
    """
    n_jobs = _n_jobs(bench_case)
    if n_jobs < 5:
        return 100
    elif n_jobs <= 20:
        return 30
    else:
        return 15


def filter_py_spy_stderr(stderr: str) -> tuple[str, bool]:
    lines = stderr.splitlines()
    filtered_lines = [
        line
        for line in lines
        if line.strip() != PY_SPY_NO_CHILD_PROCESS_ERROR
    ]
    return "\n".join(filtered_lines).strip(), len(filtered_lines) != len(lines)


def generate_runner_command(
    bench_case: Case,
    case_file: Path,
    n_runs: int,
    output_jsonl: Path,
    py_spy_output: Path | None = None,
    cprofile_output: Path | None = None,
) -> list[str]:
    command_prefix: list[str] = []
    if bench_case.bench.taskset is not None:
        command_prefix.extend(["taskset", "-c", str(bench_case.bench.taskset)])

    runner_command = [
        sys.executable,
        "-m",
        bench_case.runner_module,
        "--case-file",
        str(case_file),
        "--n-runs",
        str(n_runs),
        "--output-jsonl",
        str(output_jsonl),
    ]

    if cprofile_output is not None:
        runner_command += ["--cprofile-output", str(cprofile_output)]

    if py_spy_output is not None:
        native_flag = ["--native"] if bench_case.bench.py_spy_native else []
        return command_prefix + [
            "py-spy",
            "record",
            *native_flag,
            "--rate",
            str(py_spy_rate(bench_case)),
            "--format",
            "raw",
            "-o",
            str(py_spy_output),
            "--",
        ] + runner_command

    return command_prefix + runner_command


def parse_runner_jsonl(output_jsonl: Path) -> list[dict]:
    rows = []
    if not output_jsonl.exists():
        return rows
    with output_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_runner_from_case(
    bench_case: Case,
    py_spy_output: Path | None = None,
    cprofile_output: Path | None = None,
    n_runs_override: int | None = None,
) -> tuple[int, list[dict], dict | None]:
    bench_case_dict = bench_case.json_dict()
    n_runs = n_runs_override if n_runs_override is not None else bench_case.bench.n_runs
    bench_time_limit = bench_case.bench.time_limit
    with tempfile.TemporaryDirectory(prefix="sklbench-run-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        case_file = tmp_path / "case.json"
        output_jsonl = tmp_path / "result.jsonl"
        with case_file.open("w", encoding="utf-8") as fp:
            json.dump(bench_case_dict, fp)

        command = generate_runner_command(
            bench_case,
            case_file,
            n_runs,
            output_jsonl,
            py_spy_output,
            cprofile_output,
        )
        try:
            result = sp.run(
                command,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                encoding="utf-8",
                timeout=bench_time_limit,
            )
            return_code = result.returncode
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
        except sp.TimeoutExpired as exc:
            return_code = -9
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            timeout_message = f"Runner exceeded time limit ({bench_time_limit} seconds)."
            stderr = f"{stderr}\n{timeout_message}".strip()

        if py_spy_output is not None:
            stderr, filtered_py_spy_error = filter_py_spy_stderr(stderr)
            if return_code != 0 and filtered_py_spy_error and not stderr:
                return_code = 0

        rows = parse_runner_jsonl(output_jsonl)
        logs = {"stdout": stdout, "stderr": stderr}
        failed_case = None
        if return_code != 0:
            failed_case = {
                "case": bench_case_dict,
                "return_code": return_code,
                "command": command,
                "logs": logs,
            }
        for row in rows:
            row["logs"] = logs.copy()
        return return_code, rows, failed_case
