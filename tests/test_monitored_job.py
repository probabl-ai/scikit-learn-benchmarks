"""The job-name resolver shared by the AWS readiness handshake.

`scripts/ci/monitored-job.sh` turns a run's job list into one status, or into a
named failure. These tests drive it through a stand-in for `gh`, so the payload
shapes are explicit: one exact match, no match, two matches, a name that only
appears on a later page, and an unreadable job list.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "ci"
RESOLVER = SCRIPTS / "monitored-job.sh"
CLEAR_STALE = SCRIPTS / "clear-stale-runner-ready.sh"
AWS_PROVISION = SCRIPTS / "aws-provision.sh"

# The name the resolver defines in `scripts/ci/monitored-job.sh`; nothing else
# supplies it, so a change there is a change here.
JOB_NAME = "runner-ready"


def _install_gh(tmp_path: Path) -> Path:
    """A stand-in for `gh`: it prints the payload pages named by the environment.

    `FAKE_GH_PAGES` is a `:`-separated list of files, one per page, printed in
    order the way `gh api --paginate` writes them. `FAKE_GH_FAIL` makes every
    call fail, `FAKE_GH_LOG` records the arguments of every call.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ -n "${FAKE_GH_LOG:-}" ]; then printf "%s\\n" "$*" >>"${FAKE_GH_LOG}"; fi\n'
        'if [ -n "${FAKE_GH_FAIL:-}" ]; then echo "gh: HTTP 502" >&2; exit 1; fi\n'
        'IFS=":" read -r -a pages <<<"${FAKE_GH_PAGES:?}"\n'
        'for page in "${pages[@]}"; do cat "$page"; done\n'
    )
    gh.chmod(0o755)
    return bin_dir


def _install_aws(tmp_path: Path) -> Path:
    """A stand-in for the AWS CLI: the instance is always already running."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    aws = bin_dir / "aws"
    aws.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "ec2" ] && [ "${2:-}" = "describe-instances" ]; then\n'
        "  echo running\n"
        "fi\n"
        "exit 0\n"
    )
    aws.chmod(0o755)
    return bin_dir


def _page(tmp_path: Path, page: int, jobs) -> Path:
    """Write one jobs payload page; `jobs` is a list of (name, status) pairs."""
    payload = {
        "total_count": len(jobs),
        "jobs": [
            {"id": page * 1000 + index, "name": name, "status": status}
            for index, (name, status) in enumerate(jobs)
        ],
    }
    path = tmp_path / f"page{page}.json"
    path.write_text(json.dumps(payload))
    return path


def _run(script: Path, bin_dir: Path, pages, **overrides):
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "probabl-ai/scikit-learn-benchmarks",
        "GITHUB_RUN_ID": "17000000000",
        "GH_TOKEN": "token",
        "FAKE_GH_PAGES": ":".join(str(page) for page in pages),
    }
    env.update({key: value for key, value in overrides.items() if value is not None})
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env, cwd=REPO_ROOT
    )


@pytest.fixture
def bin_dir(tmp_path):
    return _install_gh(tmp_path)


def test_single_match_prints_its_status(tmp_path, bin_dir):
    pages = [_page(tmp_path, 1, [(JOB_NAME, "in_progress"), ("benchmark-aws", "queued")])]

    result = _run(RESOLVER, bin_dir, pages)

    assert result.returncode == 0
    assert result.stdout == "in_progress\n"


def test_absent_job_names_the_job_and_the_run(tmp_path, bin_dir):
    pages = [_page(tmp_path, 1, [("benchmark-aws", "queued"), ("cleanup-aws", "completed")])]

    result = _run(RESOLVER, bin_dir, pages)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "::error::" in result.stderr
    assert JOB_NAME in result.stderr
    assert "17000000000" in result.stderr
    assert "no job" in result.stderr


def test_two_jobs_sharing_the_name_are_an_error(tmp_path, bin_dir):
    pages = [_page(tmp_path, 1, [(JOB_NAME, "queued"), (JOB_NAME, "in_progress")])]

    result = _run(RESOLVER, bin_dir, pages)

    assert result.returncode == 3
    assert result.stdout == ""
    assert "2 jobs are named 'runner-ready'" in result.stderr
    assert "17000000000" in result.stderr


def test_a_matrix_leg_does_not_match_the_plain_name(tmp_path, bin_dir):
    pages = [_page(tmp_path, 1, [(f"{JOB_NAME} (val)", "in_progress")])]

    result = _run(RESOLVER, bin_dir, pages)

    assert result.returncode == 2
    assert JOB_NAME in result.stderr


def test_match_on_a_later_page_is_found(tmp_path, bin_dir):
    first = _page(tmp_path, 1, [(f"filler-{index}", "completed") for index in range(100)])
    second = _page(tmp_path, 2, [(JOB_NAME, "queued")])

    result = _run(RESOLVER, bin_dir, [first, second])

    assert result.returncode == 0
    assert result.stdout == "queued\n"


def test_unreadable_job_list_is_reported(tmp_path, bin_dir):
    pages = [_page(tmp_path, 1, [(JOB_NAME, "in_progress")])]

    result = _run(RESOLVER, bin_dir, pages, FAKE_GH_FAIL="1")

    assert result.returncode == 4
    assert "::error::" in result.stderr
    assert JOB_NAME in result.stderr


def test_cleanup_cancels_when_the_job_is_absent(tmp_path, bin_dir):
    pages = [_page(tmp_path, 1, [("benchmark-aws", "queued")])]
    log = tmp_path / "gh.log"

    result = _run(CLEAR_STALE, bin_dir, pages, FAKE_GH_LOG=str(log))

    assert result.returncode == 0
    assert "still missing" in result.stdout
    assert "cancel" in log.read_text()


def test_cleanup_leaves_a_picked_up_job_alone(tmp_path, bin_dir):
    pages = [_page(tmp_path, 1, [(JOB_NAME, "in_progress")])]
    log = tmp_path / "gh.log"

    result = _run(CLEAR_STALE, bin_dir, pages, FAKE_GH_LOG=str(log))

    assert result.returncode == 0
    assert "nothing to clear" in result.stdout
    assert not log.exists() or "cancel" not in log.read_text()


def test_waiter_returns_once_the_job_is_picked_up(tmp_path):
    bin_dir = _install_aws(tmp_path)
    _install_gh(tmp_path)
    pages = [_page(tmp_path, 1, [(JOB_NAME, "in_progress")])]

    result = _run(
        AWS_PROVISION,
        bin_dir,
        pages,
        INSTANCE_ID="i-0123456789abcdef0",
        REGION="eu-west-3",
        AWS_RUNNER_NAME="aws-nvidia-l4-1",
        AWS_RUNNER_GROUP="aws-gpu-runners",
    )

    assert result.returncode == 0
    assert "the readiness job is in_progress" in result.stdout


def test_waiter_fails_on_duplicate_names(tmp_path):
    bin_dir = _install_aws(tmp_path)
    _install_gh(tmp_path)
    pages = [_page(tmp_path, 1, [(JOB_NAME, "queued"), (JOB_NAME, "queued")])]

    result = _run(
        AWS_PROVISION,
        bin_dir,
        pages,
        INSTANCE_ID="i-0123456789abcdef0",
        REGION="eu-west-3",
        AWS_RUNNER_NAME="aws-nvidia-l4-1",
        AWS_RUNNER_GROUP="aws-gpu-runners",
    )

    assert result.returncode == 1
    assert "2 jobs are named 'runner-ready'" in result.stderr
