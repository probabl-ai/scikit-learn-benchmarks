import json
import subprocess as sp
import sys
import tempfile
from pathlib import Path

from ..config import Case


RUNNER_MODULES = {
    "estimator": "sklbench.runners.estimator",
    "pipeline": "sklbench.runners.pipeline",
}
PY_SPY_NO_CHILD_PROCESS_ERROR = "Error: No child process (os error 10)"


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

    if py_spy_output is not None:
        return command_prefix + [
            "py-spy",
            "record",
            "--native",
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
