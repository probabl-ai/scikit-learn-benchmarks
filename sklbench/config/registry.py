from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def to_repo_relative(path: str | Path) -> str:
    """Repo-relative, POSIX-style path for a config script, used to stamp
    `metadata.source_config` (see `load_cases_from_script`) and to key
    `dashboards/gen_*.py`'s `SOURCE_CONFIGS`. Falls back to the given path
    unchanged for anything outside the repo (e.g. a config passed by
    absolute path from elsewhere)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
