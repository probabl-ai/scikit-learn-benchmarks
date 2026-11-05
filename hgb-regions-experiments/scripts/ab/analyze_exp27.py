"""Exp 27: main and the retune against #34935: ratios, threads used, geomeans with a bootstrap over the
processes, separately on the 15 hgb_scalability cases and the 15 new shapes, the worst cases, and msd in seconds.

usage: analyze_exp27.py results.jsonl...
"""
import json, random, statistics as st, sys
from collections import defaultdict

REF, MAIN, NEW = "c325pr:main", "c325main:main", "c325retune:main"
rng = random.Random(0)


def gm(fits, cases, v, sample=False):
    pick = (lambda x: rng.choices(x, k=len(x))) if sample else (lambda x: x)
    return st.geometric_mean(st.median(pick(fits[c, v])) / st.median(pick(fits[c, REF])) for c in cases)


for path in sys.argv[1:]:
    fits, used = defaultdict(list), defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            fits[r["case"], r["variant"]].append(st.median(r["fit_s"]))
            used[r["case"]][r["variant"]] = r["phases"]["n_threads"]
    variants = [v for v in (MAIN, NEW) if any(k[1] == v for k in fits)]
    cases = [c for c in dict.fromkeys(k[0] for k in fits) if all((c, v) in fits for v in variants + [REF])]
    print(f"\n### {path.split('/')[-1].removesuffix('.jsonl')} ({len(cases)} cases, {min(len(fits[c, NEW]) for c in cases)} to {max(len(fits[c, NEW]) for c in cases)} processes)\n")
    print("| case | #34935 (T, s) | " + " | ".join(("main" if v == MAIN else "retune (T, range)") for v in variants) + " |")
    print("|" + " --- |" * (len(variants) + 2))
    for c in cases:
        ref = st.median(fits[c, REF])
        cells = []
        for v in variants:
            cells.append(f"{st.median(fits[c, v]) / ref:.2f}" if v == MAIN else f"{st.median(fits[c, v]) / ref:.2f} ({used[c][v]}, {min(fits[c, v]) / ref:.2f}-{max(fits[c, v]) / ref:.2f})")
        print(f"| {c} | {used[c][REF]}, {ref:.3g} | " + " | ".join(cells) + " |")
    for lab, group in (("original", [c for c in cases if not c.startswith("t-")]), ("new shapes", [c for c in cases if c.startswith("t-")])):
        if not group:
            continue
        parts = []
        for v in variants:
            boot = sorted(gm(fits, group, v, True) for _ in range(1000))
            parts.append(f"{'main' if v == MAIN else 'retune'} {gm(fits, group, v):.2f} ({boot[50]:.2f}-{boot[950]:.2f})")
        worst = sorted(group, key=lambda c: st.median(fits[c, NEW]) / st.median(fits[c, REF]))[-3:][::-1]
        txt = ", ".join(f"{c} {st.median(fits[c, NEW]) / st.median(fits[c, REF]):.2f} ({min(fits[c, NEW]) / st.median(fits[c, REF]):.2f}-{max(fits[c, NEW]) / st.median(fits[c, REF]):.2f})" for c in worst)
        print(f"\n{lab} ({len(group)}): " + ", ".join(parts) + f". Worst: {txt}")
        if MAIN in variants:
            vs_main = {c: st.median(fits[c, NEW]) / st.median(fits[c, MAIN]) for c in group}
            top = sorted(group, key=vs_main.get)[-3:][::-1]
            print(f"retune / main: geomean {st.geometric_mean(vs_main.values()):.2f}. Worst: " + ", ".join(f"{c} {vs_main[c]:.2f}" for c in top))
    msd = "year_prediction_msd"
    if (msd, NEW) in fits and (msd, REF) in fits:
        print(f"\nmsd: " + ", ".join(f"{lab} {st.median(fits[msd, v]):.2f} s" for lab, v in (("main", MAIN), ("#34935", REF), ("retune", NEW)) if (msd, v) in fits))
