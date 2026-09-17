#!/usr/bin/env python3
"""Regenerate dashboard HTML when benchmark inputs or reporting code changes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


WATCH_ROOTS = (Path("results"), Path("sklbench/reporting"), Path("dashboards"))

# Real result payloads are tens of KB or more; an unfetched Git LFS pointer
# stub is on the order of 130 bytes. Cheap size filter before reading content.
LFS_POINTER_MAX_SIZE = 512
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch results/, sklbench/reporting/, and dashboards/ and regenerate all "
            "dashboard pages when a change is detected."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_site"),
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
    if path.parent == Path("dashboards") and path.suffix == ".html":
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


def _lfs_pointer_files() -> list[Path]:
    results_root = Path("results")
    if not results_root.exists():
        return []
    pointers = []
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > LFS_POINTER_MAX_SIZE:
                continue
            with open(path, "rb") as f:
                head = f.read(len(LFS_POINTER_PREFIX))
        except OSError:
            continue
        if head == LFS_POINTER_PREFIX:
            pointers.append(path)
    return pointers


def _pull_lfs_pointers(paths: list[Path]) -> bool:
    if not paths:
        return True
    print(
        f"Found {len(paths)} unfetched Git LFS file(s) under results/; "
        "running `git lfs pull`...",
        flush=True,
    )
    for path in paths[:10]:
        print(f"  {path}")
    if len(paths) > 10:
        print(f"  ... and {len(paths) - 10} more")

    command = [
        "git", "lfs", "pull",
        "--exclude", "",
        "--include", ",".join(str(path) for path in paths),
    ]
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        print(
            "`git lfs pull` failed; dashboard generation may error on LFS "
            "pointer files (see CONTRIBUTING.md > Previewing Dashboards "
            "Locally).",
            file=sys.stderr,
        )
        return False
    return True


def _generate(output_dir: Path) -> bool:
    _pull_lfs_pointers(_lfs_pointer_files())

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Regenerating dashboards into {output_dir}...", flush=True)
    command = [sys.executable, "dashboards/index.py", "--output-dir", str(output_dir)]
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        print(f"dashboards/index.py failed with exit code {result.returncode}.", file=sys.stderr)
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
        "Watching results/, sklbench/reporting/, and dashboards/. "
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

            if _generate(args.output_dir):
                previous = _snapshot()
            else:
                print(
                    "Generation failed; will retry on the next detected change.",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
