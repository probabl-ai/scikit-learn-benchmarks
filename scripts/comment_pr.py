"""
Create or update a GitHub PR comment via the REST API directly (urllib,
stdlib only), keyed on a hidden HTML marker so repeated calls for the same
PR (e.g. on every push) edit one comment in place instead of posting a new
one each time.

This repo's org restricts workflows to an allowlist of GitHub Actions, and
PR-comment-automation actions aren't on it, so this goes through the API
directly - same rationale as scripts/create_pr.py.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_ROOT = "https://api.github.com"


def _request(method: str, url: str, token: str, data: dict | None = None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        sys.stderr.write(exc.read().decode("utf-8", errors="replace") + "\n")
        raise


def find_existing_comment(repo: str, pr_number: int, marker: str, token: str):
    page = 1
    while True:
        url = f"{API_ROOT}/repos/{repo}/issues/{pr_number}/comments?per_page=100&page={page}"
        comments = _request("GET", url, token)
        if not comments:
            return None
        for comment in comments:
            if marker in comment.get("body", ""):
                return comment["id"]
        if len(comments) < 100:
            return None
        page += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument(
        "--marker",
        required=True,
        help="Hidden HTML-comment marker embedded in the comment body, used "
        "to find and update an existing comment on later calls. Use a "
        "distinct marker per bot/flow so they don't clobber each other's "
        "comments on the same PR.",
    )
    parser.add_argument("--body-file", required=True)
    args = parser.parse_args()

    token = os.environ["GITHUB_TOKEN"]
    body = args.marker + "\n" + open(args.body_file, encoding="utf-8").read()

    existing_id = find_existing_comment(args.repo, args.pr_number, args.marker, token)
    if existing_id is not None:
        result = _request(
            "PATCH",
            f"{API_ROOT}/repos/{args.repo}/issues/comments/{existing_id}",
            token,
            data={"body": body},
        )
    else:
        result = _request(
            "POST",
            f"{API_ROOT}/repos/{args.repo}/issues/{args.pr_number}/comments",
            token,
            data={"body": body},
        )
    print(result["html_url"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
