"""
Parse the `sklbench-compare` directive out of a PR description, for the
PR-comparison benchmark workflow (.github/workflows/pr-comparison.yml).

Looks for a fenced code block:

    ```sklbench-compare
    sklearn_pr: https://github.com/scikit-learn/scikit-learn/pull/12345
    config: configs/hgb_scaling.py
    runners: intel-laptop, intel-gnr
    ```

and validates its `key: value` lines. Writes outcomes to $GITHUB_OUTPUT
rather than communicating via exit code, so the calling workflow can branch
on step outputs (directive_present, ok, and either the parsed fields or an
error message) without needing `continue-on-error`.
"""

import argparse
import json
import re
import secrets
import sys

FENCE_RE = re.compile(
    r"^```sklbench-compare[ \t]*\r?\n(.*?)\r?\n```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
SKLEARN_PR_RE = re.compile(
    r"^(?:https?://github\.com/scikit-learn/scikit-learn/pull/)?(\d+)/?$"
)
CONFIG_RE = re.compile(r"^configs/[A-Za-z0-9_./-]+\.py$")
VALID_RUNNERS = {"intel-laptop", "intel-gnr"}
REQUIRED_KEYS = {"sklearn_pr", "config"}
KNOWN_KEYS = REQUIRED_KEYS | {"runners"}


class Malformed(Exception):
    pass


def parse_fields(block: str) -> dict:
    fields = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise Malformed(f"line has no ':' separator: {line!r}")
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in KNOWN_KEYS:
            raise Malformed(f"unknown key {key!r} (expected one of {sorted(KNOWN_KEYS)})")
        if key in fields:
            raise Malformed(f"key {key!r} is repeated")
        fields[key] = value

    missing = REQUIRED_KEYS - fields.keys()
    if missing:
        raise Malformed(f"missing required key(s): {sorted(missing)}")

    sklearn_pr_match = SKLEARN_PR_RE.match(fields["sklearn_pr"])
    if sklearn_pr_match is None:
        raise Malformed(
            f"sklearn_pr {fields['sklearn_pr']!r} is not a "
            "scikit-learn/scikit-learn PR URL or number"
        )
    sklearn_pr_number = sklearn_pr_match.group(1)

    if not CONFIG_RE.match(fields["config"]) or ".." in fields["config"].split("/"):
        raise Malformed(f"config {fields['config']!r} must look like 'configs/<name>.py'")

    runners_raw = fields.get("runners", "").strip()
    if not runners_raw:
        runners = ["intel-laptop", "intel-gnr"]
    else:
        tokens = [t for t in re.split(r"[,\s]+", runners_raw.lower()) if t]
        if not tokens:
            raise Malformed("runners key is present but empty")
        runners = []
        for token in tokens:
            if token == "both":
                names = ("intel-laptop", "intel-gnr")
            elif token in VALID_RUNNERS:
                names = (token,)
            else:
                raise Malformed(
                    f"unknown runner {token!r} (expected intel-laptop, intel-gnr, or both)"
                )
            for name in names:
                if name not in runners:
                    runners.append(name)

    return {
        "sklearn_pr_number": sklearn_pr_number,
        "config": fields["config"],
        "runners": runners,
    }


def write_output(path: str, key: str, value: str) -> None:
    if "\n" in value:
        delimiter = f"__EOF_{secrets.token_hex(8)}__"
        block = f"{key}<<{delimiter}\n{value}\n{delimiter}\n"
    else:
        block = f"{key}={value}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--github-output", required=True)
    args = parser.parse_args()

    body = open(args.body_file, encoding="utf-8").read()

    match = FENCE_RE.search(body)
    if match is None:
        write_output(args.github_output, "directive_present", "false")
        write_output(args.github_output, "ok", "false")
        return 0

    write_output(args.github_output, "directive_present", "true")
    try:
        parsed = parse_fields(match.group(1))
    except Malformed as exc:
        write_output(args.github_output, "ok", "false")
        write_output(args.github_output, "error", str(exc))
        return 0

    write_output(args.github_output, "ok", "true")
    write_output(args.github_output, "sklearn_pr_number", parsed["sklearn_pr_number"])
    write_output(args.github_output, "config", parsed["config"])
    write_output(args.github_output, "runners", json.dumps(parsed["runners"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
