"""Exp 25: each variant against #34935 on the validation shapes: ratios, geomean with a bootstrap
over the processes, and the worst cases.

usage: analyze_exp25.py results.jsonl...
"""
import json, random, statistics as st, sys
from collections import defaultdict

REF = "c325knob:main"
rng = random.Random(0)


def label(v):
    if v == "c325main:main":
        return "main"
    if v == REF:
        return "#34935"
    if "ASTAIR" in v and "100000000:32" in v:
        return "A64"
    if "ASTAIR" in v:
        return "A16" if "20000000:16" in v else "A32"
    if "PSTAIR" in v:
        return "P16" if "50000000:8" in v else "P8"
    return v.replace("c325knob:main~", "")


for path in sys.argv[1:]:
    fits, used = defaultdict(list), defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            fits[r["case"], r["variant"]].append(st.median(r["fit_s"]))
            used[r["case"]][r["variant"]] = r["phases"]["n_threads"]
    variants = [v for v in dict.fromkeys(k[1] for k in fits) if v != REF]
    variants.sort(key=lambda v: (v != "c325main:main", v))
    cases = [c for c in dict.fromkeys(k[0] for k in fits) if all((c, v) in fits for v in variants + [REF])]
    print(f"\n### {path.split('/')[-1].removesuffix('.jsonl')} ({len(cases)} cases)\n")
    print("| case | #34935 (T, s) | " + " | ".join(label(v) + " (T)" for v in variants) + " | #34935 spread |")
    print("|" + " --- |" * (len(variants) + 3))
    for c in cases:
        ref = st.median(fits[c, REF])
        cells = " | ".join(f"{st.median(fits[c, v]) / ref:.2f} ({used[c][v]})" for v in variants)
        print(f"| {c} | {used[c][REF]}, {ref:.3g} | {cells} | {(max(fits[c, REF]) - min(fits[c, REF])) / ref:.0%} |")
    row = []
    for v in variants:
        boot = sorted(st.geometric_mean(st.median(rng.choices(fits[c, v], k=len(fits[c, v]))) / st.median(rng.choices(fits[c, REF], k=len(fits[c, REF]))) for c in cases) for _ in range(1000))
        row.append(f"{st.geometric_mean(st.median(fits[c, v]) / st.median(fits[c, REF]) for c in cases):.2f} ({boot[50]:.2f}-{boot[950]:.2f})")
    print("| geomean | | " + " | ".join(row) + " | |")
    for v in variants:
        if v != "c325main:main":
            worst = sorted(cases, key=lambda c: st.median(fits[c, v]) / st.median(fits[c, REF]))[-3:][::-1]
            txt = ", ".join(f"{c} {st.median(fits[c, v]) / st.median(fits[c, REF]):.2f} ({min(fits[c, v]) / st.median(fits[c, REF]):.2f}-{max(fits[c, v]) / st.median(fits[c, REF]):.2f})" for c in worst)
            print(f"\nworst {label(v)}: {txt}")
