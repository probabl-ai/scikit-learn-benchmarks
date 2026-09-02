import gzip
import hashlib
import json
import logging
import math
import re
import shlex
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Case, EstimatorCase, PipelineCase
from .commands import run_runner_from_case
from .env import get_environment_info

logger = logging.getLogger(__name__)
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_LOG_CHARS = 4000


def hash_from_json_repr(x: Any, hash_limit: int = 5) -> str:
    h = hashlib.sha256()
    h.update(bytes(json.dumps(x), encoding="utf-8"))
    return h.hexdigest()[:hash_limit]


def get_hardware_hash(hardware_info: dict) -> str:
    return hash_from_json_repr(hardware_info, hash_limit=6)


def get_software_hash(software_info: dict) -> str:
    return hash_from_json_repr(software_info, hash_limit=6)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _case_basename(bench_case: Case) -> str:
    case_slug = bench_case.name(shortened=True, separator="_")
    case_slug = _UNSAFE_FILENAME_CHARS.sub("_", case_slug).strip("_")
    case_hash = hash_from_json_repr(bench_case.json_dict())
    return f"{case_slug}_{case_hash}_{_timestamp()}"


def _truncate_log(log: str) -> str:
    if len(log) <= _MAX_LOG_CHARS:
        return log
    return f"... truncated ...\n{log[-_MAX_LOG_CHARS:]}"


def _extract_shape(rows: list[dict] | None) -> tuple[int | None, int | None]:
    if not rows:
        return None, None
    data_desc = rows[0].get("data_desc")
    if not isinstance(data_desc, dict):
        return None, None
    if "n_samples" in data_desc:
        return data_desc.get("n_samples"), data_desc.get("n_features")
    fit_desc = data_desc.get("fit")
    if isinstance(fit_desc, dict):
        return fit_desc.get("samples"), fit_desc.get("features")
    return None, None


def _print_progress_line(
    index: int,
    total: int,
    case_name: str,
    duration_s: float,
    n_samples: int | None,
    n_features: int | None,
) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    shape = (
        f"{n_samples} x {n_features}"
        if n_samples is not None and n_features is not None
        else "? x ?"
    )
    print(
        f"[{timestamp}] {index}/{total} {case_name} {shape} - took {duration_s:.0f}s",
        file=sys.stderr,
    )


def _warmup(duration_s: float = 30) -> None:
    import numpy as np

    print(f"[sklbench] Warming up ({duration_s:.0f}s matmul)...", file=sys.stderr)
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        np.matmul(a, b)


def _gzip_file(source: Path, destination: Path) -> None:
    destination.write_bytes(
        gzip.compress(source.read_bytes(), compresslevel=9, mtime=0)
    )


def _py_spy_timeout(normal_run_duration: float, n_runs: int) -> float:
    """Time budget for the py-spy pass, sized off how long the un-profiled
    run actually took rather than the case's full `time_limit`.

    py-spy's ptrace-based sampling can occasionally make a run pathologically
    slow (see `py_spy_rate`'s docstring); reusing the full `time_limit` as
    the timeout let those cases run nearly as long as the untimed benchmark
    even though the py-spy pass itself only repeats ~n_runs/3 times. Scaling
    off the observed `normal_run_duration` keeps the timeout tied to reality,
    with a `1/sqrt(n_runs)` margin that shrinks as more repeats make that
    duration a more stable estimate, plus a flat 2s floor for interpreter/
    import startup on very fast cases.
    """
    return 2 + normal_run_duration * (1 + 1 / math.sqrt(n_runs))


def _run_cprofile_pass(
    bench_case: Case, basename: str, profiles_dir: Path
) -> tuple[int, dict | None]:
    """One reduced-n_runs cProfile pass, gzipped to `results/profiles/`.

    Shares the n_runs reduction with the py-spy pass so both profilers are
    measuring a comparable slice of the case.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    cprofile_path = profiles_dir / f"{basename}.prof.gz"
    cprofile_n_runs = max(1, bench_case.bench.n_runs // 3)
    with tempfile.TemporaryDirectory(prefix="sklbench-cprofile-") as cprofile_tmp_dir:
        raw_cprofile_path = Path(cprofile_tmp_dir) / f"{basename}.prof"
        cprofile_return_code, _, cprofile_failed_case = run_runner_from_case(
            bench_case,
            cprofile_output=raw_cprofile_path,
            n_runs_override=cprofile_n_runs,
        )
        if cprofile_return_code == 0:
            _gzip_file(raw_cprofile_path, cprofile_path)
    return cprofile_return_code, cprofile_failed_case


def _log_failed_case(
    bench_case: Case,
    failed_case: dict | None,
    *,
    stage: str,
    return_code: int,
) -> None:
    case_name = bench_case.name(shortened=True)
    message_parts = [
        f"{stage} failed for {case_name!r} with return code {return_code}."
    ]
    if failed_case is not None:
        error = failed_case.get("error")
        if error:
            message_parts.append(f"error: {error}")
        command = failed_case.get("command")
        if command:
            message_parts.append(
                "command: " + shlex.join(str(part) for part in command)
            )
        logs = failed_case.get("logs", {})
        stderr = logs.get("stderr")
        if stderr:
            message_parts.append(f"stderr:\n{_truncate_log(stderr)}")
        stdout = logs.get("stdout")
        if stdout:
            message_parts.append(f"stdout:\n{_truncate_log(stdout)}")
    logger.warning("\n".join(message_parts))


def save_environment_sidecars(
    hardware_hash: str,
    software_hash: str,
    env_info: dict,
    results_dir: str,
):
    results_root = Path(results_dir)
    hardware_env_dir = results_root / "hardware-envs"
    software_env_dir = results_root / "software-envs"
    hardware_env_dir.mkdir(parents=True, exist_ok=True)
    software_env_dir.mkdir(parents=True, exist_ok=True)

    env_files = [
        (hardware_env_dir / f"{hardware_hash}.json", env_info["hardware"]),
        (software_env_dir / f"{software_hash}.json", env_info["software"]),
    ]
    for env_file, env_content in env_files:
        try:
            with env_file.open("x", encoding="utf-8") as fp:
                json.dump(env_content, fp, indent=4)
        except FileExistsError:
            pass


def save_benchmark_record(
    record_path: Path,
    bench_case: Case,
    rows: list[dict],
    failed_case: dict | None,
    hardware_hash: str,
    software_hash: str,
):
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "hardware_hash": hardware_hash,
        "software_hash": software_hash,
        "case": bench_case.json_dict(),
        "results": rows,
        "failed_case": failed_case,
    }
    with record_path.open("x", encoding="utf-8") as fp:
        json.dump(record, fp, indent=4)


def _load_case_dataset(bench_case: Case) -> tuple[int | None, int | None]:
    if isinstance(bench_case, EstimatorCase):
        from ..runners.datasets import load_raw_data as load_data

        raw_data, _ = load_data(bench_case)
        x = raw_data.get("x", raw_data.get("x_train"))
    elif isinstance(bench_case, PipelineCase):
        from ..runners.pipeline import load_data

        x, _ = load_data(bench_case)
    else:
        raise TypeError(f"Unsupported case type: {type(bench_case)!r}")
    if x is None:
        return None, None
    n_features = x.shape[1] if len(x.shape) == 2 else None
    return x.shape[0], n_features


def load_datasets_only(bench_cases: list[Case], args) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    return_code = 0
    n_cases = len(bench_cases)
    for index, bench_case in enumerate(bench_cases, start=1):
        case_name = bench_case.name(shortened=True)
        case_start = time.monotonic()
        n_samples, n_features = None, None
        try:
            n_samples, n_features = _load_case_dataset(bench_case)
        except Exception as exc:
            return_code = -1
            logger.warning(f"Failed to load dataset for {case_name!r}: {exc!r}")
            if args.exit_on_error:
                break
        finally:
            _print_progress_line(
                index,
                n_cases,
                case_name,
                time.monotonic() - case_start,
                n_samples,
                n_features,
            )
    return return_code


def orchestrate_benchmarks(
    bench_cases: list[Case],
    args,
) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    if any("test" not in config for config in args.config):
        _warmup()

    env_info = get_environment_info()
    hardware_hash = get_hardware_hash(env_info["hardware"])
    software_hash = get_software_hash(env_info["software"])
    save_environment_sidecars(
        hardware_hash,
        software_hash,
        env_info,
        args.results_dir,
    )

    return_code = 0
    results_root = Path(args.results_dir)
    records_dir = results_root / "records"
    profiles_dir = results_root / "profiles"

    n_cases = len(bench_cases)
    for index, bench_case in enumerate(bench_cases, start=1):
        basename = _case_basename(bench_case)
        record_path = records_dir / f"{basename}.json"
        profile_path = profiles_dir / f"{basename}.raw.gz"
        record_saved = False
        cprofile_already_run = False
        case_name = bench_case.name(shortened=True)
        case_start = time.monotonic()
        rows = None
        try:
            normal_run_start = time.monotonic()
            bench_return_code, rows, failed_case = run_runner_from_case(bench_case)
            normal_run_duration = time.monotonic() - normal_run_start
            save_benchmark_record(
                record_path,
                bench_case,
                rows,
                failed_case,
                hardware_hash,
                software_hash,
            )
            record_saved = True
            if bench_return_code != 0:
                return_code = bench_return_code
                _log_failed_case(
                    bench_case,
                    failed_case,
                    stage="Benchmark",
                    return_code=bench_return_code,
                )
                if args.exit_on_error:
                    break
            if bench_return_code == 0 and bench_case.bench.py_spy_profiling:
                profiles_dir.mkdir(parents=True, exist_ok=True)
                profile_n_runs = max(1, bench_case.bench.n_runs // 3)
                with tempfile.TemporaryDirectory(
                    prefix="sklbench-profile-"
                ) as profile_tmp_dir:
                    raw_profile_path = Path(profile_tmp_dir) / f"{basename}.raw"
                    profile_return_code, _, profile_failed_case = run_runner_from_case(
                        bench_case,
                        py_spy_output=raw_profile_path,
                        n_runs_override=profile_n_runs,
                        timeout_override=_py_spy_timeout(
                            normal_run_duration, bench_case.bench.n_runs
                        ),
                    )
                    if profile_return_code == 0:
                        _gzip_file(raw_profile_path, profile_path)

                if profile_return_code == -9 and not bench_case.bench.cprofile_profiling:
                    # py-spy timed out - cProfile doesn't share its ptrace/
                    # scheduler-churn failure modes, so fall back to it for
                    # this case instead of losing the profile entirely.
                    _log_failed_case(
                        bench_case,
                        profile_failed_case,
                        stage="Profiling benchmark (py-spy timed out, falling back to cProfile)",
                        return_code=profile_return_code,
                    )
                    profile_return_code, profile_failed_case = _run_cprofile_pass(
                        bench_case, basename, profiles_dir
                    )
                    cprofile_already_run = True

                if profile_return_code != 0:
                    return_code = profile_return_code
                    _log_failed_case(
                        bench_case,
                        profile_failed_case,
                        stage="Profiling benchmark",
                        return_code=profile_return_code,
                    )
                    if args.exit_on_error:
                        break
            if (
                bench_return_code == 0
                and bench_case.bench.cprofile_profiling
                and not cprofile_already_run
            ):
                cprofile_return_code, cprofile_failed_case = _run_cprofile_pass(
                    bench_case, basename, profiles_dir
                )
                if cprofile_return_code != 0:
                    return_code = cprofile_return_code
                    _log_failed_case(
                        bench_case,
                        cprofile_failed_case,
                        stage="cProfile profiling benchmark",
                        return_code=cprofile_return_code,
                    )
                    if args.exit_on_error:
                        break
        except KeyboardInterrupt:
            return_code = -1
            failed_case = {
                "case": bench_case.json_dict(),
                "return_code": return_code,
                "error": "KeyboardInterrupt",
                "logs": {"stdout": "", "stderr": "KeyboardInterrupt"},
            }
            if not record_saved:
                save_benchmark_record(
                    record_path,
                    bench_case,
                    [],
                    failed_case,
                    hardware_hash,
                    software_hash,
                )
            _log_failed_case(
                bench_case,
                failed_case,
                stage="Benchmark",
                return_code=return_code,
            )
            break
        except Exception as exc:
            return_code = -1
            failed_case = {
                "case": bench_case.json_dict(),
                "return_code": return_code,
                "error": repr(exc),
                "logs": {"stdout": "", "stderr": str(exc)},
            }
            if not record_saved:
                save_benchmark_record(
                    record_path,
                    bench_case,
                    [],
                    failed_case,
                    hardware_hash,
                    software_hash,
                )
            _log_failed_case(
                bench_case,
                failed_case,
                stage="Benchmark setup",
                return_code=return_code,
            )
            if args.exit_on_error:
                break
        finally:
            n_samples, n_features = _extract_shape(rows)
            _print_progress_line(
                index,
                n_cases,
                case_name,
                time.monotonic() - case_start,
                n_samples,
                n_features,
            )
    return return_code
