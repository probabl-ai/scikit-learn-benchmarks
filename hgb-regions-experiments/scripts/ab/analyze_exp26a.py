"""Exp 26a: active caps 16, 32, 64 per shape: fit against cap 16, threads used, and process range.
usage: analyze_exp26a.py results.jsonl..."""
import json, statistics as st, sys
from collections import defaultdict

for path in sys.argv[1:]:
    fits, used = defaultdict(list), defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            v = int(r["variant"].rsplit(":", 1)[1])
            fits[r["case"], v].append(st.median(r["fit_s"]))
            used[r["case"]][v] = r["phases"]["n_threads"]
    print(f"\n### {path.split('/')[-1].removesuffix('.jsonl')}\n")
    print("| case | cap 16 (T, s) | cap 32 (T) | cap 64 (T) |")
    print("| --- | --- | --- | --- |")
    for c in dict.fromkeys(k[0] for k in fits):
        if not all((c, v) in fits for v in (16, 32, 64)):
            continue
        b = st.median(fits[c, 16])
        cell = lambda v: f"{st.median(fits[c, v]) / b:.2f} ({used[c][v]}, {min(fits[c, v]) / b:.2f}-{max(fits[c, v]) / b:.2f})"
        print(f"| {c} | {used[c][16]}, {b:.3g} | {cell(32)} | {cell(64)} |")
