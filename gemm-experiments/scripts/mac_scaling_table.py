"""Thread-scaling table from a sweep.py CSV: per variant and shape, the
median over rounds (and max over rounds) of gflops_median for each thread
count, plus the per-round ratio to the same variant's T=1 in that round,
which cancels the P-cluster clock changes between rounds.

    python scripts/mac_scaling_table.py results/mac-m2/e8_scaling.csv [ROUNDS]

ROUNDS (e.g. 3,4) keeps only those rounds.
"""

import csv
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
if len(sys.argv) > 2:
    keep = sys.argv[2].split(",")
    rows = [r for r in rows if r["round"] in keep]
v = defaultdict(dict)
shapes, variants, threads = [], [], []
for r in rows:
    s = f"{r['dtype']} {r['m']}x{r['n']}x{r['k']}:{r['trans']}"
    t = int(r["threads_requested"])
    for lst, x in ((shapes, s), (variants, r["variant"]), (threads, t)):
        if x not in lst:
            lst.append(x)
    v[r["variant"], s, t][int(r["round"])] = float(r["gflops_median"])
for var in variants:
    print(f"\n== {var}: GF/s median over rounds (max) | speed-up vs T=1, median of per-round ratios")
    print(f"{'shape':<24}" + "".join(f"{'T' + str(t):>11}" for t in threads))
    for s in shapes:
        line = f"{s:<24}"
        for t in threads:
            d = v[var, s, t]
            line += f"{statistics.median(d.values()):>6.0f}({max(d.values()):>4.0f})"
        ratios = []
        for t in threads:
            d, d1 = v[var, s, t], v[var, s, 1]
            ratios.append(statistics.median(d[r] / d1[r] for r in d if r in d1))
        print(line + "  | " + " ".join(f"{x:4.1f}" for x in ratios))
