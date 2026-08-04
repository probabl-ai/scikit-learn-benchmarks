"""
Open a GitHub PR for a pushed bench-results branch, embedding the sklbench
log in a collapsible <details> block in the PR description.

Uses the GitHub REST API directly via urllib (stdlib only) so it doesn't
depend on `gh` or any other CLI being installed on the runner.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MAX_LOG_BYTES = 50_000


def build_body(summary: str, log_path: str) -> str:
    log_bytes = open(log_path, "rb").read()
    truncated = len(log_bytes) > MAX_LOG_BYTES
    if truncated:
        log_bytes = log_bytes[-MAX_LOG_BYTES:]
    log_text = log_bytes.decode("utf-8", errors="replace")
    note = "... (log truncated, showing the tail) ...\n" if truncated else ""
    return (
        f"{summary}\n\n"
        "<details><summary>bench.log</summary>\n\n"
        "```\n"
        f"{note}{log_text}\n"
        "```\n"
        "</details>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log", required=True, help="Path to the bench.log to embed")
    args = parser.parse_args()

    body = build_body(args.summary, args.log)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{args.repo}/pulls",
        data=json.dumps(
            {"title": args.title, "head": args.head, "base": args.base, "body": body}
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        sys.stderr.write(exc.read().decode("utf-8", errors="replace") + "\n")
        raise
    print(result["html_url"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
