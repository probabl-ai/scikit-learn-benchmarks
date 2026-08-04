"""
Build a Markdown PR body embedding the sklbench log in a collapsible
<details> block, for use with peter-evans/create-pull-request's
`body-path` input.

The log is truncated to its last MAX_LOG_BYTES bytes to stay comfortably
under GitHub's PR body size limit (65536 characters).
"""

import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log", required=True, help="Path to the bench.log to embed")
    parser.add_argument("--out", required=True, help="Path to write the PR body to")
    args = parser.parse_args()

    with open(args.out, "w") as f:
        f.write(build_body(args.summary, args.log))


if __name__ == "__main__":
    main()
