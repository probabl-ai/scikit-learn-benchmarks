"""Exp 23: guard sweep, ratio of each variant to the capped baseline, and the geomean with its spread.

usage: analyze_exp23.py [BASE_VARIANT] results.jsonl...
"""
import json, random, statistics as st, sys
from collections import defaultdict

BASE = sys.argv.pop(1) if sys.argv[1].startswith("c325") else "c325knob:main~PCAP=8"
rng = random.Random(0)
for path in sys.argv[1:]:
    fits, used = defaultdict(list), {}
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            fits[r["case"], r["variant"]].append(st.median(r["fit_s"]))
            if r["variant"] == BASE:
                used[r["case"]] = r["phases"]["n_threads"]
    cases = [c for c in dict.fromkeys(k[0] for k in fits) if (c, BASE) in fits]
    variants = [v for v in dict.fromkeys(k[1] for k in fits) if v != BASE]
    short = lambda v: v.replace(BASE + "~", "").replace("c325knob:main", "#34935")
    print(f"\n### {path.split('/')[-1].removesuffix('.jsonl')}\n")
    print("| case | T | base (s) | " + " | ".join(map(short, variants)) + " | base spread |")
    print("|" + " --- |" * (len(variants) + 4))
    for c in cases:
        b = st.median(fits[c, BASE])
        cells = " | ".join(f"{st.median(fits[c, v]) / b:.2f}" if (c, v) in fits else "" for v in variants)
        print(f"| {c} | {used[c]} | {b:.3g} | {cells} | {(max(fits[c, BASE]) - min(fits[c, BASE])) / b:.0%} |")
    gm = []
    for v in variants:
        cs = [c for c in cases if (c, v) in fits]
        boot = sorted(st.geometric_mean(st.median(rng.choices(fits[c, v], k=len(fits[c, v]))) / st.median(rng.choices(fits[c, BASE], k=len(fits[c, BASE]))) for c in cs) for _ in range(500))
        gm.append(f"{st.geometric_mean(st.median(fits[c, v]) / st.median(fits[c, BASE]) for c in cs):.2f} ({boot[25]:.2f}-{boot[475]:.2f})")
    print("| geomean | | | " + " | ".join(gm) + " | |")
