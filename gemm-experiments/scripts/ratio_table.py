"""Base/build ratio per shape and thread count from a sweep CSV (min of rounds).

    python scripts/ratio_table.py results/intel-gamer-gpu/tc4.csv [--skip-first]
"""

import csv
import sys
from collections import defaultdict

skip = "--skip-first" in sys.argv
path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
rows = list(csv.DictReader(open(path)))
variants = list(dict.fromkeys(r["variant"] for r in rows))
t = defaultdict(list)
for r in rows:
    if skip and r["round"] == "0":
        continue
    shape = r["trans"] + ":" + "x".join(d for d in (r["m"], r["n"], r["k"]) if d != "1")
    t[shape, r["variant"], int(r["threads"])].append(float(r["median_s"]))
shapes = list(dict.fromkeys(k[0] for k in t))
threads = sorted({k[2] for k in t})
a, b = variants[0], variants[-1]
print(f"{'shape':22}" + "".join(f"{x:>7}" for x in threads) + f"   ({a}/{b})")
for s in shapes:
    print(f"{s:22}" + "".join(f"{min(t[s, a, x]) / min(t[s, b, x]):7.2f}" for x in threads))
