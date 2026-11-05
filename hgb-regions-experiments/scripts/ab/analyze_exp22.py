"""Exp 22: for each case, fit time against the thread count, as a ratio to #34935's rule.

usage: analyze_exp22.py results.jsonl...
"""
import json, re, statistics as st, sys
from collections import defaultdict

REF = "c325knob:main"
for path in sys.argv[1:]:
    fits, used = defaultdict(list), {}
    failed = []
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"]:
            fits[r["case"], r["variant"]].append(st.median(r["fit_s"]))
            if r["variant"] == REF:
                used[r["case"]] = r["phases"]["n_threads"], r["phases"]["active_wait"]
        else:
            failed.append((r["case"], r["variant"], r["rc"]))
    # A profile still running can miss a variant of its last case.
    cases = [c for c in dict.fromkeys(k[0] for k in fits)
             if (c, REF) in fits and (c, "c325main:main") in fits]
    variants = list(dict.fromkeys(k[1] for k in fits))
    forced = sorted({v for v in variants if "~T=" in v},
                    key=lambda v: (int(re.search(r"T=(\d+)", v)[1]), "FS" in v))
    short = lambda v: v.split("~", 1)[1].replace("~", " ")
    fs_rule = [v for v in variants if "~FS=" in v and "~T=" not in v]
    print(f"\n### {path.split('/')[-1].removesuffix('.jsonl')}\n")
    print("| case | rule T | #34935 (s) | main | " + " | ".join(map(short, fs_rule + forced)) + " | best |")
    print("|" + " --- |" * (len(fs_rule + forced) + 5))
    gains = []
    for c in cases:
        ref = st.median(fits[c, REF])
        ratio = lambda v: st.median(fits[c, v]) / ref if (c, v) in fits else float("nan")
        best = min(forced + fs_rule + [REF], key=lambda v: ratio(v) if (c, v) in fits else 9e9)
        gains.append(ratio(best))
        cells = " | ".join(f"{ratio(v):.2f}" for v in fs_rule + forced)
        print(f"| {c} | {used[c][0]} | {ref:.3g} | {ratio('c325main:main'):.2f} | {cells} | {short(best) if '~' in best else 'rule'} {ratio(best):.2f} |")
    gm = lambda vs: st.geometric_mean(vs)
    print(f"\ngeomean: main {gm([st.median(fits[c, 'c325main:main']) / st.median(fits[c, REF]) for c in cases]):.2f}, best per case {gm(gains):.2f}")
    if failed:
        print("failed:", failed)
