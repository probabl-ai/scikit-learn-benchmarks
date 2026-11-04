"""Median times of the A/B runs, and whether the `check` values agree across builds.

usage: analyze.py FILE.jsonl...   (*-other.jsonl from other_ab.py, *-hgb.jsonl from bench_hgb.py)
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def fmt(values):
    if not values:
        return "NA"
    return f"{statistics.median(values):.3f} ({min(values):.3f}-{max(values):.3f})"


for path in sys.argv[1:]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    failed = [r for r in rows if r.get("rc") != 0]
    print(f"\n## {Path(path).name}: {len(rows)} runs, {len(failed)} failed")
    for r in failed[:5]:
        print("  FAILED", r.get("case"), r.get("build", r.get("variant")), str(r.get("stderr"))[-300:])
    times, checks = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(set))
    columns = []
    for r in rows:
        if r.get("rc") != 0:
            continue
        col = r.get("build", r.get("variant"))
        if col not in columns:
            columns.append(col)
        if "fit_s" in r:
            times[r["case"]][col].extend(r["fit_s"])
        else:
            times[r["case"]][col].append(r["time_s"])
            checks[r["case"]][col].add(r["check"])
    columns.sort()
    print("| case | " + " | ".join(columns) + " |")
    print("| --- |" + " --- |" * len(columns))
    for case, per_col in times.items():
        print(f"| {case} | " + " | ".join(fmt(per_col[c]) for c in columns) + " |")
    for case, per_col in checks.items():
        values = {v for s in per_col.values() for v in s}
        spread = (max(values) - min(values)) / max(abs(max(values)), 1e-300)
        print(f"check {case}: {len(values)} distinct values, relative spread {spread:.2e}")
