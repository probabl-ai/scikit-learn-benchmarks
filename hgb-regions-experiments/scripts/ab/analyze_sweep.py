"""Exp 16 table: median fit (s) per split_indices threshold, one row per case.

usage: analyze_sweep.py results.jsonl...
"""
import glob, json, statistics as st, sys
from collections import defaultdict

VS = ["none+fuse3"] + [f"abs+fuse3@{x}" for x in (10000, 30000, 100000, 300000, 1000000, 2000000000)] + ["pr+fuse3"]
for f in sys.argv[1:]:
    d = defaultdict(list)
    for line in open(f):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            d[r["case"], r["variant"]] += r["fit_s"]
    print("\n###", f.split("/")[-1], "\n")
    print("| case | always threaded | 10K | 30K | 100K | 300K | 1M | always serial | PR rule |\n|" + " --- |" * 9)
    for c in dict.fromkeys(k[0] for k in d):
        print(f"| {c} | " + " | ".join(f"{st.median(d[c, v]):.2f}" if (c, v) in d else "" for v in VS) + " |")
