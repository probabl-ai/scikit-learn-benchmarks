"""
Open a GitHub PR via the REST API directly (urllib, stdlib only).

This repo's org restricts workflows to an allowlist of third-party GitHub
Actions, and pull-request-automation actions aren't on it, so PR creation
has to go through the API directly rather than a dedicated action.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body-file", required=True)
    args = parser.parse_args()

    body = open(args.body_file, encoding="utf-8").read()
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
