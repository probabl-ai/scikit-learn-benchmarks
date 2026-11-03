"""Tables of the A/B results: median fit time per (case, variant), and its ratio to `pr`.

usage: python analyze.py results_*.jsonl [--markdown]
"""
import json
import statistics
import sys
from collections import defaultdict

ORDER = ["main", "pr", "abs", "pr+fuse", "pr+fuse3", "abs+fuse3", "none+fuse3"]
paths = [a for a in sys.argv[1:] if not a.startswith("--")]
md = "--markdown" in sys.argv

fits = defaultdict(list)  # (profile, case, variant) -> fit times
failed = []
for path in paths:
    for line in open(path):
        r = json.loads(line)
        if r["rc"] != 0 or not r["fit_s"]:
            failed.append((r["profile"], r["case"], r["variant"], r["rc"]))
            continue
        fits[r["profile"], r["case"], r["variant"]] += r["fit_s"]

for profile in sorted({k[0] for k in fits}):
    cases = list(dict.fromkeys(k[1] for k in fits if k[0] == profile))
    variants = [v for v in ORDER if any((profile, c, v) in fits for c in cases)]
    print(f"\n### {profile}: median fit (s), and in parentheses time / pr\n")
    head = ["case"] + variants
    rows = []
    for c in cases:
        med = {v: statistics.median(fits[profile, c, v]) for v in variants if (profile, c, v) in fits}
        ref = med.get("pr")
        row = [c]
        for v in variants:
            if v not in med:
                row.append("")
            elif v == "pr" or ref is None:
                row.append(f"{med[v]:.3g}")
            else:
                row.append(f"{med[v]:.3g} ({med[v] / ref:.2f})")
        rows.append(row)
    # geometric mean of ratios to pr over cases that have both
    gm = ["geomean / pr"]
    for v in variants:
        rs = [statistics.median(fits[profile, c, v]) / statistics.median(fits[profile, c, "pr"])
              for c in cases if (profile, c, v) in fits and (profile, c, "pr") in fits]
        gm.append(f"{statistics.geometric_mean(rs):.2f}" if rs and v != "pr" else "1")
    rows.append(gm)
    if md:
        print("| " + " | ".join(head) + " |")
        print("|" + " --- |" * len(head))
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(x[i]) for x in [head] + rows) for i in range(len(head))]
        for r in [head] + rows:
            print("  ".join(x.ljust(w[i]) for i, x in enumerate(r)))
if failed:
    print("\nfailed:", failed)
