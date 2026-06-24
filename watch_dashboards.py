#!/usr/bin/env python3
"""Regenerate dashboard HTML when benchmark inputs or reporting code changes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


WATCH_ROOTS = (Path("results"), Path("reporting"), Path("dashboard"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch results/, reporting/, and dashboard/ and regenerate all "
            "dashboard pages when a change is detected."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dashboard"),
        help="Directory where generated HTML files are written.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Generate dashboards once and exit.",
    )
    return parser.parse_args()


def _is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    if path.parent == Path("dashboard") and path.suffix == ".html":
        return True
    return False


def _snapshot() -> dict[Path, tuple[int, int]]:
    files = {}
    for root in WATCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or _is_ignored(path):
                continue
            stat = path.stat()
            files[path] = (stat.st_mtime_ns, stat.st_size)
    return files


def _changed_paths(
    before: dict[Path, tuple[int, int]],
    after: dict[Path, tuple[int, int]],
) -> list[Path]:
    paths = sorted(set(before) | set(after))
    return [path for path in paths if before.get(path) != after.get(path)]


def _generator_scripts() -> list[Path]:
    return sorted(Path("dashboard").glob("gen_*.py"))


def _generate(output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    scripts = _generator_scripts()
    if not scripts:
        print("No dashboard/gen_*.py scripts found.", file=sys.stderr)
        return False

    print(f"Regenerating {len(scripts)} dashboards into {output_dir}...", flush=True)
    for script in scripts:
        command = [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            print(f"{script} failed with exit code {result.returncode}.", file=sys.stderr)
            return False
    print("Dashboard regeneration complete.", flush=True)
    return True


def main() -> int:
    args = _parse_args()

    if args.interval <= 0:
        print("--interval must be positive.", file=sys.stderr)
        return 2

    if args.once:
        return 0 if _generate(args.output_dir) else 1

    _generate(args.output_dir)
    previous = _snapshot()
    print(
        "Watching results/, reporting/, and dashboard/. "
        "Press Ctrl-C to stop.",
    )

    try:
        while True:
            time.sleep(args.interval)
            current = _snapshot()
            changed = _changed_paths(previous, current)
            if not changed:
                continue

            print("Detected changes:")
            for path in changed[:10]:
                print(f"  {path}")
            if len(changed) > 10:
                print(f"  ... and {len(changed) - 10} more")

            _generate(args.output_dir)
            previous = _snapshot()
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
