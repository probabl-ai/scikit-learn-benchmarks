"""Best-of-rounds table from a run_sklearn_e2e.sh CSV: base, build, ratio per case and thread count."""

import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
labels = list(dict.fromkeys(r["label"] for r in rows))
best = defaultdict(lambda: float("inf"))
iters = defaultdict(set)
for r in rows:
    k = (r["case"], int(r["threads"]), r["label"])
    best[k] = min(best[k], float(r["best_s"]))
    iters[r["case"]].add(r.get("n_iter", ""))
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
