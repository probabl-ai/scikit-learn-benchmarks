"""
Resolve a scikit-learn PR number to its source fork's owner and branch, via
the GitHub REST API directly (urllib, stdlib only) - same rationale as
scripts/create_pr.py: this org restricts workflows to an allowlist of
GitHub Actions, so there's no dedicated action for this either.

Outputs just `owner`, matching run.sh's `env@owner:ref` shorthand, which
hardcodes the fork's repo name as "scikit-learn" and reconstructs the
remote as `https://github.com/<owner>/scikit-learn.git` itself. Since a
PR's fork isn't guaranteed to be named "scikit-learn", this explicitly
checks the fork's repo name and fails clearly (ok=false) rather than
silently handing run.sh an owner whose repo it can't actually find.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_ROOT = "https://api.github.com"


def write_output(path: str, key: str, value: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--repo", default="scikit-learn/scikit-learn")
    parser.add_argument("--github-output", required=True)
    args = parser.parse_args()

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        f"{API_ROOT}/repos/{args.repo}/pulls/{args.pr_number}",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request) as response:
            pr = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            write_output(args.github_output, "ok", "false")
            write_output(
                args.github_output,
                "error",
                f"{args.repo} PR #{args.pr_number} not found",
            )
            return 0
        sys.stderr.write(exc.read().decode("utf-8", errors="replace") + "\n")
        raise

    head = pr.get("head") or {}
    fork_repo = head.get("repo")
    if not fork_repo:
        write_output(args.github_output, "ok", "false")
        write_output(
            args.github_output,
            "error",
            f"the source fork for {args.repo} PR #{args.pr_number} no longer "
            "exists (deleted fork)",
        )
        return 0

    if fork_repo["name"] != "scikit-learn":
        write_output(args.github_output, "ok", "false")
        write_output(
            args.github_output,
            "error",
            f"the source fork for {args.repo} PR #{args.pr_number} is named "
            f"{fork_repo['name']!r}, not 'scikit-learn' - not supported, since "
            "this flow runs the comparison via run.sh's env@owner:ref "
            "shorthand, which assumes the fork repo is named scikit-learn",
        )
        return 0

    write_output(args.github_output, "ok", "true")
    write_output(args.github_output, "owner", fork_repo["owner"]["login"])
    write_output(args.github_output, "head_ref", head["ref"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
