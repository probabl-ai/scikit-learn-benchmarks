"""Median fit and grower phase times per (profile, case, variant), per fit, with the min-max of the fit.

usage: analyze_phases.py results.jsonl... [--ref VARIANT]
"""
import json, statistics as st, sys
from collections import defaultdict

args = sys.argv[1:]
ref = args[args.index("--ref") + 1] if "--ref" in args else None
paths = [a for i, a in enumerate(args) if a.endswith(".jsonl")]
rows = defaultdict(list)
for path in paths:
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            rows[r["profile"], r["case"], r["variant"]].append(r)
for profile, case in dict.fromkeys(k[:2] for k in rows):
    print(f"\n### {profile} {case}\n")
    print("| variant | n | fit med (min-max) | ratio | hist | find | apply | other | idle entries (M, shallow/deep) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    variants = [k[2] for k in rows if k[:2] == (profile, case)]
    med_of = lambda v: st.median(f for r in rows[profile, case, v] for f in r["fit_s"])
    r0 = med_of(ref) if ref in variants else None
    for v in sorted(variants, key=med_of):
        rs = rows[profile, case, v]
        fits = [f for r in rs for f in r["fit_s"]]
        def phase(p):
            vals = [r["phases"][p] / max(1, len(r["fit_s"])) for r in rs if "phases" in r]
            return st.median(vals) if vals else float("nan")
        hist, find, apply, grow = (phase(p) for p in ("hist", "find", "apply", "grow"))
        ratio = f"{st.median(fits) / r0:.2f}" if r0 else ""
        idle = [r["phases"].get("idle_entries") for r in rs if "phases" in r]
        idle_txt = ""
        if idle and idle[0]:
            shallow = st.median(sum(i.get(s, 0) for s in ("state0", "state1")) for i in idle) / 1e6
            deep = st.median(sum(v for s, v in i.items() if s not in ("state0", "state1")) for i in idle) / 1e6
            idle_txt = f"{shallow:.2f} / {deep:.2f}"
        print(f"| {v} | {len(fits)} | {st.median(fits):.2f} ({min(fits):.2f}-{max(fits):.2f}) | {ratio} | {hist:.2f} | {find:.2f} | {apply:.2f} | {st.median(fits) - hist - find - apply:.2f} | {idle_txt} |")
