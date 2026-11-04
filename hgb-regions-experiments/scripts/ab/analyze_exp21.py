"""Exp 21: msd in seconds, and the cases where hgb/merge-regions is furthest from #34935 and main."""
import json
import statistics as st
import sys
from collections import defaultdict

MAIN, PR, CLEAN = "c325main:main", "c325pr:pr", "c325clean:pr"
for path in sys.argv[1:]:
    fits = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        fits[r["case"], r["variant"]].append(st.median(r["fit_s"]))
    cases = list(dict.fromkeys(k[0] for k in fits))
    med = lambda c, v: st.median(fits[c, v])
    ratio = {c: med(c, CLEAN) / med(c, PR) for c in cases}
    vs_main = {c: med(c, CLEAN) / med(c, MAIN) for c in cases}
    name = path.split("/")[-1].removesuffix(".jsonl").removeprefix("exp21-")
    msd = "year_prediction_msd"
    print(f"{name}: msd main {med(msd, MAIN):.3g} #34935 {med(msd, PR):.3g} #42 {med(msd, CLEAN):.3g}")
    fmt = lambda d: ", ".join(f"{c} {d[c]:.2f}" for c in sorted(d, key=d.get, reverse=True)[:4])
    print(f"  #42/#34935 worst: {fmt(ratio)}")
    print(f"  #42/main worst: {fmt(vs_main)}")
    c = "L-stumps"
    print(f"  L-stumps #34935 {sorted(round(x / med(c, MAIN), 2) for x in fits[c, PR])} #42 {sorted(round(x / med(c, MAIN), 2) for x in fits[c, CLEAN])}")
