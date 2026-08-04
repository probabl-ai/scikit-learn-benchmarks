"""
Classify a captured `run.sh` log into a pass/fail verdict per Pixi
environment.

sklbench's orchestrator catches per-case failures itself (logging a
"... failed for '...' with return code N" WARNING and moving on), so a few
isolated case failures are expected and not a sign anything is actually
broken. A session is only considered failed when either:

- the orchestrator loop did not reach its last case (nothing else stops the
  loop early, so this means it crashed or was killed outside its per-case
  handling), or
- more than half of that environment's cases hit a hard crash (a negative
  return code: killed by its own per-case time limit, OOM, or another
  signal, as opposed to a graceful/soft failure raised by the case itself).

Reads a combined stdout+stderr log as produced by `run.sh`, which delimits
each environment with a "=== pixi run -e <env> ... ===" line. Prints one
verdict per environment found and exits non-zero if any environment failed.
"""

import argparse
import re
import sys

ENV_HEADER_RE = re.compile(r"^=== pixi run -e (\S+) .* ===\s*$")
PROGRESS_RE = re.compile(r"^\[sklbench\] (\d+)/(\d+) ")
FAILURE_RE = re.compile(
    r"WARNING - sklbench\.orchestrator\.implementation - "
    r"(?:Benchmark|Profiling benchmark|Benchmark setup) failed for '[^']*' "
    r"with return code (-?\d+)\."
)

HARD_CRASH_FRACTION_THRESHOLD = 0.5


def split_by_env(log_text: str) -> dict[str, list[str]]:
    sessions: dict[str, list[str]] = {}
    current_env = None
    for line in log_text.splitlines():
        header = ENV_HEADER_RE.match(line)
        if header:
            current_env = header.group(1)
            sessions.setdefault(current_env, [])
            continue
        if current_env is not None:
            sessions[current_env].append(line)
    return sessions or {"session": log_text.splitlines()}


def classify_session(lines: list[str]) -> tuple[bool, str]:
    total = current = None
    for line in lines:
        match = PROGRESS_RE.search(line)
        if match:
            current, total = int(match.group(1)), int(match.group(2))

    if total is None:
        return False, "no progress output found (crashed before running any case)"
    if current != total:
        return False, f"orchestrator stopped mid-run ({current}/{total} cases reached)"

    hard_crashes = sum(
        1 for line in lines if (m := FAILURE_RE.search(line)) and int(m.group(1)) < 0
    )
    if hard_crashes / total > HARD_CRASH_FRACTION_THRESHOLD:
        return False, f"{hard_crashes}/{total} cases hit a hard crash"

    return True, f"completed {total}/{total} cases ({hard_crashes} hard crashes)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Combined run.sh log to classify (defaults to stdin).",
    )
    args = parser.parse_args()

    all_ok = True
    for env, lines in split_by_env(args.log.read()).items():
        ok, reason = classify_session(lines)
        all_ok &= ok
        print(f"{'OK' if ok else 'FAILED'}: {env}: {reason}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
