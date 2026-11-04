"""Table from a run_sklearn_e2e.sh CSV: base, build, ratio per case and thread count.

Each cell is the best of rounds, or with --median the median of rounds (more
robust on a laptop, where single runs can be 15% fast or slow). --skip-first
drops round 1: the first process after an idle period runs faster on a laptop.
"""

import csv
import statistics
import sys
from collections import defaultdict

median = "--median" in sys.argv
skip = "--skip-first" in sys.argv
path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
rows = list(csv.DictReader(open(path)))
labels = list(dict.fromkeys(r["label"] for r in rows))
rows = [r for r in rows if not (skip and r["round"] == "1")]
vals = defaultdict(list)
iters = defaultdict(set)
for r in rows:
    vals[r["case"], int(r["threads"]), r["label"]].append(float(r["best_s"]))
    iters[r["case"]].add(r.get("n_iter") or "")
best = {k: statistics.median(v) if median else min(v) for k, v in vals.items()}
threads = sorted({int(r["threads"]) for r in rows})
cases = list(dict.fromkeys(r["case"] for r in rows))
a, b = labels[0], labels[-1]
print(f"{'case':26}" + "".join(f"{t:>22}" for t in threads) + "   n_iter")
for c in cases:
    cells = []
    for t in threads:
        x, y = best[c, t, a] * 1e3, best[c, t, b] * 1e3
        cells.append(f"{x:7.1f} {y:7.1f} {x / y:5.2f}x")
    print(f"{c:26}" + "".join(f"{s:>22}" for s in cells) + "   " + ",".join(sorted(iters[c])))
print(f"cells: {a} ms, {b} ms, {a}/{b}")
