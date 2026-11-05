"""Exp 26: passive cap of 8 against 16, per shape: median fit with process range, and threads used.

usage: analyze_exp26.py results.jsonl...
"""
import json, statistics as st, sys
from collections import defaultdict

for path in sys.argv[1:]:
    fits, used = defaultdict(list), defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            v = "#34935" if "~" not in r["variant"] else r["variant"].rsplit(":", 1)[1]
            fits[r["case"], v].append(st.median(r["fit_s"]))
            used[r["case"]][v] = r["phases"]["n_threads"]
    print(f"\n### {path.split('/')[-1].removesuffix('.jsonl')}\n")
    print("| case | #34935 (T, s) | cap 8 (T, s) | cap 16 (T, s) | 16 / 8 (range) |")
    print("| --- | --- | --- | --- | --- |")
    for c in dict.fromkeys(k[0] for k in fits):
        if (c, "16") not in fits or (c, "8") not in fits or (c, "#34935") not in fits:
            continue
        m = {v: st.median(fits[c, v]) for v in ("#34935", "8", "16")}
        lo, hi = min(fits[c, "16"]) / m["8"], max(fits[c, "16"]) / m["8"]
        print(f"| {c} | {used[c]['#34935']}, {m['#34935']:.3g} | {used[c]['8']}, {m['8']:.3g} | {used[c]['16']}, {m['16']:.3g} | {m['16'] / m['8']:.2f} ({lo:.2f}-{hi:.2f}) |")
