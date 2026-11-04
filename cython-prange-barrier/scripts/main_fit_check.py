"""Median fit and grower phases per case and build, and whether the models agree across builds.

usage: main_fit_check.py FILE.jsonl...
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PHASES = ("hist_time", "find_split_time", "apply_split_time", "grower_init_time", "binning_time")
SAME = (("metrics", "predict", "R2"), ("metrics", "fit", "R2"), ("attributes", "n_iter"),
        ("attributes", "tree_n_threads"))


def get(row, path):
    for key in path:
        row = row.get(key, {})
    return row if row != {} else None


for path in sys.argv[1:]:
    fits, phases, same = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(set)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        key = (r["case"], r["variant"].split(":")[0])
        fits[key].extend(r["fit_s"])
        for row in r["rows"]:
            for name in PHASES:
                value = get(row, ("attributes", name))
                if value is not None:
                    phases[key][name].append(value)
            for field in SAME:
                same[(r["case"], ".".join(field))].add(get(row, field))
    print(f"\n## {Path(path).name}")
    print("| case | build | n | fit | range | " + " | ".join(p.replace("_time", "") for p in PHASES) + " |")
    print("| --- | --- | --- | --- | --- |" + " --- |" * len(PHASES))
    for key, values in sorted(fits.items()):
        cols = [f"{statistics.median(phases[key][p]):.2f}" if phases[key][p] else "NA" for p in PHASES]
        print(f"| {key[0]} | {key[1]} | {len(values)} | {statistics.median(values):.2f} "
              f"| {min(values):.2f}-{max(values):.2f} | " + " | ".join(cols) + " |")
    for (case, field), values in sorted(same.items()):
        values = sorted(v for v in values if v is not None)
        if not values:
            continue
        spread = max(values) - min(values)
        print(f"{case} {field}: {len(values)} distinct, spread {spread:.2e}")
