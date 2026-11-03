"""Per-routine thread scaling table from a run_routines.sh CSV.

For each shape, prints the median time per thread count for every variant,
the thread count where the first variant is fastest (each cell is the best
round's median, so a round disturbed by another process doesn't count), and two ratios at the
largest thread count: first variant vs its own best, and vs the last variant.

    python scripts/routines_table.py results/xeon-s1/routines.csv
"""

import csv
import sys
from collections import defaultdict


def fmt(t):
    return f"{t * 1e6:.0f}u" if t < 1e-3 else f"{t * 1e3:.1f}m" if t < 1 else f"{t:.2f}s"


rows = list(csv.DictReader(open(sys.argv[1])))
times = defaultdict(list)
for r in rows:
    shape = r["trans"] + ":" + "x".join(
        d for d in (r["m"], r["n"], r["k"]) if d != "1"
    )
    times[shape, r["variant"], int(r["threads"])].append(float(r["median_s"]))
variants = list(dict.fromkeys(r["variant"] for r in rows))
threads = sorted({int(r["threads"]) for r in rows})
shapes = list(dict.fromkeys(k[0] for k in times))
first, last, tmax = variants[0], variants[-1], threads[-1]
print(f"{'shape':22} {'variant':10} " + " ".join(f"{t:>7}" for t in threads)
      + f"  best_T  {tmax}/best  {first}/{last}@{tmax}")
for s in shapes:
    for v in variants:
        med = {t: min(times[s, v, t]) for t in threads if times[s, v, t]}
        line = f"{s:22} {v:10} " + " ".join(f"{fmt(med[t]) if t in med else '-':>7}" for t in threads)
        if med and tmax in med:
            best = min(med, key=med.get)
            line += f"  {best:>6}  {med[tmax] / med[best]:7.2f}"
            if v == first:
                other = times[s, last, tmax]
                if other:
                    line += f"  {med[tmax] / min(other):8.2f}"
        print(line)
