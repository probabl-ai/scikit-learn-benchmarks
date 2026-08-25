"""
Parse the `sklbench-compare` directive out of a PR description, for the
PR-comparison benchmark workflow (.github/workflows/pr-comparison.yml).

Looks for a fenced code block:

    ```sklbench-compare
    sklearn_ref: cakedev0:ridge/optim_cholesky
    config: configs/hgb_scaling.py
    runners: intel-laptop, intel-gnr
    ```

`sklearn_ref` is the same `owner:ref` shorthand run.sh's `env@owner:ref`
spec takes (see CONTRIBUTING.md) - the fork owner and the branch/ref to
benchmark on it, taken straight from the PR author instead of resolved from
a scikit-learn PR number via the GitHub API. That's a no-op simplification
trust-wise, not a new risk class: this directive already only runs for
OWNER/MEMBER/COLLABORATOR authors, who already have workflow_dispatch
access to run-benchmarks.yml with equally arbitrary env@owner:ref input.

Validates its `key: value` lines and writes outcomes to $GITHUB_OUTPUT
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
# Mirrors GitHub's own username rules (alnum and single hyphens, no
# leading/trailing hyphen, max 39 chars) - not a security boundary (see
# module docstring), just catches typos with a clear error early.
GITHUB_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
CONFIG_RE = re.compile(r"^configs/[A-Za-z0-9_./-]+\.py$")
VALID_RUNNERS = {"intel-laptop", "intel-gnr"}
REQUIRED_KEYS = {"sklearn_ref", "config"}
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

    sklearn_ref = fields["sklearn_ref"]
    if ":" not in sklearn_ref:
        raise Malformed(f"sklearn_ref {sklearn_ref!r} must look like 'owner:ref'")
    sklearn_owner, sklearn_head_ref = sklearn_ref.split(":", 1)
    if not sklearn_owner or not sklearn_head_ref:
        raise Malformed(f"sklearn_ref {sklearn_ref!r} must look like 'owner:ref'")
    if not GITHUB_OWNER_RE.match(sklearn_owner):
        raise Malformed(
            f"sklearn_ref owner {sklearn_owner!r} is not a valid GitHub username"
        )
    if sklearn_head_ref.startswith("-") or any(c.isspace() for c in sklearn_head_ref):
        raise Malformed(f"sklearn_ref ref {sklearn_head_ref!r} is not a valid git ref")

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
        "sklearn_owner": sklearn_owner,
        "sklearn_head_ref": sklearn_head_ref,
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
    write_output(args.github_output, "sklearn_owner", parsed["sklearn_owner"])
    write_output(args.github_output, "sklearn_head_ref", parsed["sklearn_head_ref"])
    write_output(args.github_output, "config", parsed["config"])
    write_output(args.github_output, "runners", json.dumps(parsed["runners"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
