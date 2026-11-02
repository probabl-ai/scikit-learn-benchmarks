"""Where benchmark records come from: the CI run that produced them, or the
commit (and pull request) that added them to the repository.

Records written by a GitHub Actions run carry its URL (`ci_run_url`, see
`sklbench.orchestrator.implementation.save_benchmark_record`). Older records
and those from manual runs only have the git history: the commit that added
the file, and the pull request that merged it into the checked-out branch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
import subprocess
from typing import Iterable

RESULTS_REPOSITORY_URL = "https://github.com/probabl-ai/scikit-learn-benchmarks"

_MERGE_PR_RE = re.compile(r"^Merge pull request #(\d+) from (\S+)")
_SQUASH_PR_RE = re.compile(r"\(#(\d+)\)$")
# Branch pushed by .github/workflows/benchmark-run.yml.
_BENCH_BRANCH_RE = re.compile(r"/bench-results/.+-(\d+)$")


@dataclass(frozen=True)
class _Commit:
    sha: str
    parents: tuple[str, ...]
    author: str
    subject: str


@dataclass(frozen=True)
class _CommitOrigin:
    pr_number: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class RecordSource:
    run_url: str | None = None
    commit_sha: str | None = None
    commit_author: str | None = None
    commit_subject: str | None = None
    pr_number: str | None = None

    @property
    def from_ci(self) -> bool:
        # Older CI runs did not record their URL in the results.
        return self.run_url is not None or self.commit_author == "github-actions[bot]"

    @property
    def commit_url(self) -> str | None:
        if self.commit_sha is None:
            return None
        return f"{RESULTS_REPOSITORY_URL}/commit/{self.commit_sha}"

    @property
    def pr_url(self) -> str | None:
        if self.pr_number is None:
            return None
        return f"{RESULTS_REPOSITORY_URL}/pull/{self.pr_number}"


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_commits(repo: Path) -> dict[str, _Commit]:
    out = _git(repo, "log", "--format=%H%x00%P%x00%an%x00%s%x00")
    commits = {}
    for entry in (out or "").split("\x00\n"):
        if not entry.strip():
            continue
        sha, parents, author, subject = entry.strip("\n").split("\x00")
        commits[sha] = _Commit(sha, tuple(parents.split()), author, subject)
    return commits


def _ancestors(commits: dict[str, _Commit], sha: str) -> set[str]:
    seen: set[str] = set()
    stack = [sha]
    while stack:
        sha = stack.pop()
        if sha in seen or sha not in commits:
            continue
        seen.add(sha)
        stack.extend(commits[sha].parents)
    return seen


def _commit_origins(commits: dict[str, _Commit]) -> dict[str, _CommitOrigin]:
    """The pull request (and benchmark run) that merged each commit.

    A PR merge commit brings in the commits reachable from its second parent
    but not from its first. When several PR merges bring in the same commit
    (a PR merged into another PR's branch), the innermost one wins.
    """
    merged_by: dict[str, tuple[int, _CommitOrigin]] = {}
    for commit in commits.values():
        squash = _SQUASH_PR_RE.search(commit.subject)
        if squash:
            merged_by[commit.sha] = (0, _CommitOrigin(squash.group(1)))
            continue
        merge = _MERGE_PR_RE.match(commit.subject)
        if not merge or len(commit.parents) < 2:
            continue
        branch_run = _BENCH_BRANCH_RE.search(merge.group(2))
        origin = _CommitOrigin(merge.group(1), branch_run and branch_run.group(1))
        brought_in = _ancestors(commits, commit.parents[1]) - _ancestors(
            commits, commit.parents[0]
        )
        for sha in brought_in | {commit.sha}:
            if sha not in merged_by or len(brought_in) < merged_by[sha][0]:
                merged_by[sha] = (len(brought_in), origin)
    return {sha: origin for sha, (_, origin) in merged_by.items()}


@lru_cache(maxsize=None)
def _added_by_commit(records_dir: Path) -> tuple[dict[Path, _Commit], dict[str, _CommitOrigin]]:
    """Map each record file to the commit that added it, plus commit origins."""
    top = _git(records_dir, "rev-parse", "--show-toplevel")
    if top is None:
        return {}, {}
    repo = Path(top.strip())
    commits = _read_commits(repo)
    # When HEAD's history is cut short (shallow clone), its oldest fetched
    # commit seems to add every file.
    shallow_file = repo / (_git(repo, "rev-parse", "--git-path", "shallow") or "").strip()
    if shallow_file.is_file() and set(shallow_file.read_text().split()) & commits.keys():
        return {}, {}
    # --no-renames: a file moved or rewritten under a new name was added by
    # the commit that did so, not by the one that added the old name.
    out = _git(
        repo, "log", "--no-renames", "--diff-filter=A",
        "--format=%x00%H", "--name-only", "--", str(records_dir),
    )
    added: dict[Path, _Commit] = {}
    sha = None
    for line in (out or "").splitlines():
        if line.startswith("\x00"):
            sha = line[1:]
        elif line and sha in commits:
            # Newest first: keep the latest commit that added this path.
            added.setdefault((repo / line).resolve(), commits[sha])
    return added, _commit_origins(commits)


def record_source(record_path: Path | None, ci_run_url: str | None) -> RecordSource:
    if record_path is None:
        return RecordSource(run_url=ci_run_url)
    record_path = Path(record_path).resolve()
    added, origins = _added_by_commit(record_path.parent)
    commit = added.get(record_path)
    if commit is None:
        return RecordSource(run_url=ci_run_url)
    origin = origins.get(commit.sha, _CommitOrigin())
    if ci_run_url is None and origin.run_id is not None:
        ci_run_url = f"{RESULTS_REPOSITORY_URL}/actions/runs/{origin.run_id}"
    return RecordSource(
        run_url=ci_run_url,
        commit_sha=commit.sha,
        commit_author=commit.author,
        commit_subject=commit.subject,
        pr_number=origin.pr_number,
    )


def summarize_sources(
    records: Iterable[tuple[Path | None, str | None, datetime, str | None]],
) -> list[dict]:
    """One row per source, newest first, from `(record_path, ci_run_url,
    timestamp, source_config)` tuples (one per record)."""
    groups: dict[RecordSource, list] = defaultdict(list)
    for record_path, ci_run_url, timestamp, source_config in records:
        groups[record_source(record_path, ci_run_url)].append((timestamp, source_config))
    rows = []
    for source, entries in groups.items():
        timestamps = [timestamp for timestamp, _ in entries]
        rows.append({
            "source": source,
            "count": len(entries),
            "start": min(timestamps),
            "end": max(timestamps),
            "configs": sorted({config for _, config in entries if config}),
        })
    rows.sort(key=lambda row: row["end"], reverse=True)
    return rows
