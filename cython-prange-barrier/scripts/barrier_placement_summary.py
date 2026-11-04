"""Median of each metric per runtime, work size and barrier, from barrier_placement.sh output.

usage: barrier_placement_summary.py LOG
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ("wall_us", "loop_mean_us", "loop_max_us", "entry_max_us", "stacked_threads", "regions_stacked", "moved_threads")
rows = defaultdict(lambda: defaultdict(list))
for line in Path(sys.argv[1]).read_text().splitlines():
    fields = dict(re.findall(r"(\w+)=([\w.]+)", line))
    if "wall_us" not in fields:
        continue
    key = (fields.get("cfg", "-"), fields["rt"], int(fields["n"]), int(fields["barrier"]))
    for m in METRICS:
        rows[key][m].append(float(fields[m]))
print("| cfg | runtime | n | barrier | runs | " + " | ".join(METRICS) + " |")
print("| --- | --- | --- | --- | --- |" + " --- |" * len(METRICS))
for key in sorted(rows):
    cols = [f"{statistics.median(rows[key][m]):.2f}" for m in METRICS]
    print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(rows[key]['wall_us'])} | " + " | ".join(cols) + " |")
