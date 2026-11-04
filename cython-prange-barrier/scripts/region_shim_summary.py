"""Median of the region_shim totals per config and build, then the sites with the largest gap between builds.

usage: region_shim_summary.py DIR   (files are NAME-BUILD-REP.txt)
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TOTAL = ("n", "wall", "gap", "wake", "spread", "work_mean", "work_max", "exit", "team", "stacked", "moved")
SITE = ("n", "wall", "wake", "spread", "work_mean", "work_max", "exit", "stacked", "moved")
totals = defaultdict(lambda: defaultdict(list))
sites = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for path in sorted(Path(sys.argv[1]).glob("*.txt")):
    name, build, _ = path.stem.rsplit("-", 2)
    for line in path.read_text().splitlines():
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        if fields["site"] == "TOTAL":
            for m in TOTAL:
                totals[(name, build)][m].append(float(fields[m]))
        else:
            for m in SITE:
                sites[name][fields["site"]][(build, m)].append(float(fields[m]))


def med(values):
    return statistics.median(values) if values else float("nan")


print("| config | build | runs | " + " | ".join(TOTAL) + " |")
print("| --- | --- | --- |" + " --- |" * len(TOTAL))
for key in sorted(totals):
    cols = [f"{med(totals[key][m]):.2f}" for m in TOTAL]
    print(f"| {key[0]} | {key[1]} | {len(totals[key]['wall'])} | " + " | ".join(cols) + " |")

for name in sites:
    print(f"\n## {name}: sites by wall, m331 / mB2 (the addresses differ between builds, so sorted by wall, matched by rank)")
    for build in ("m331", "mB2"):
        ranked = sorted(sites[name], key=lambda s: -med(sites[name][s][(build, "wall")]) if sites[name][s][(build, "wall")] else 0)[:8]
        print(f"-- {build}")
        for s in ranked:
            d = sites[name][s]
            if not d[(build, "wall")]:
                continue
            print(s[:34].ljust(34), " ".join(f"{m}={med(d[(build, m)]):.2f}" for m in SITE))
