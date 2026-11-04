"""Exp 18 tables: median fit (min-max) per case and variant, ratio to `pr`, and the noise.

The geomean's interval comes from a bootstrap over the processes of each cell.
usage: analyze_noise.py [--ref VARIANT] results.jsonl...
"""
import json
import statistics as st
import sys
from collections import defaultdict

import random

args = sys.argv[1:]
REF = "pr"
if args[0] == "--ref":
    REF, args = args[1], args[2:]
ORDER = ["c325main:main", "c325pr:pr", "c325clean:pr", "pr", "pr+fuse3", "abs+fuse3", "clean:pr"]
rng = random.Random(0)
for path in args:
    fits = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            fits[r["case"], r["variant"]].append(st.median(r["fit_s"]))
    cases = list(dict.fromkeys(k[0] for k in fits))
    variants = [v for v in ORDER if any((c, v) in fits for c in cases)]
    print(f"\n### {path.split('/')[-1].removesuffix('.jsonl')}\n")
    print("| case | " + " | ".join(variants) + " | ref spread |")
    print("|" + " --- |" * (len(variants) + 2))
    for c in cases:
        ref = st.median(fits[c, REF])
        cells = []
        for v in variants:
            med = st.median(fits[c, v])
            if v == REF:
                cells.append(f"{med:.3g}")
            else:
                cells.append(f"{med / ref:.2f} ({min(fits[c, v]) / ref:.2f}-{max(fits[c, v]) / ref:.2f})")
        spread = (max(fits[c, REF]) - min(fits[c, REF])) / ref
        print(f"| {c} | " + " | ".join(cells) + f" | {spread:.0%} |")
    gm = []
    for v in variants:
        if v == REF:
            gm.append("1")
            continue
        boot = []
        for _ in range(2000):
            boot.append(st.geometric_mean(
                st.median(rng.choices(fits[c, v], k=len(fits[c, v])))
                / st.median(rng.choices(fits[c, REF], k=len(fits[c, REF])))
                for c in cases
            ))
        boot.sort()
        point = st.geometric_mean(st.median(fits[c, v]) / st.median(fits[c, REF]) for c in cases)
        gm.append(f"{point:.2f} ({boot[50]:.2f}-{boot[1950]:.2f})")
    print(f"| geomean / {REF} | " + " | ".join(gm) + " | |")
