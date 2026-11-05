"""Exp 26b: active caps 16, 20, 24 between 20M and 50M, against cap 16. usage: analyze_exp26b.py results.jsonl..."""
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
    print("| case | cap 16 (T, s) | cap 20 (T) | cap 24 (T) |\n| --- | --- | --- | --- |")
    ratios = {20: [], 24: []}
    for c in dict.fromkeys(k[0] for k in fits):
        if not all((c, v) in fits for v in (16, 20, 24)):
            continue
        b = st.median(fits[c, 16])
        for v in ratios:
            ratios[v].append(st.median(fits[c, v]) / b)
        cell = lambda v: f"{st.median(fits[c, v]) / b:.2f} ({used[c][v]}, {min(fits[c, v]) / b:.2f}-{max(fits[c, v]) / b:.2f})"
        print(f"| {c} | {used[c][16]}, {b:.3g} | {cell(20)} | {cell(24)} |")
    print(f"| geomean | | {st.geometric_mean(ratios[20]):.2f} | {st.geometric_mean(ratios[24]):.2f} |")
