"""Pivot a sweep.py CSV: median over rounds of gflops_median per
(dtype, shape) x (variant, threads), plus the max round-to-round drift.

    python scripts/mac_summarize.py results/mac-m2/e1_square.csv [--cols threads]
"""

import argparse
import csv
import statistics
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--metric", default="gflops_median")
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    vals = defaultdict(list)
    cols, shapes = [], []
    for r in rows:
        shape = f"{r['dtype']} {r['m']}x{r['n']}x{r['k']}:{r['trans']}"
        col = f"{r['variant']}/T{r['threads_requested']}"
        if col not in cols:
            cols.append(col)
        if shape not in shapes:
            shapes.append(shape)
        vals[shape, col].append(float(r[args.metric]))
    w = max(len(c) for c in cols) + 1
    print(f"{'shape':<28}" + "".join(f"{c:>{w}}" for c in cols))
    drift = []
    for s in shapes:
        line = f"{s:<28}"
        for c in cols:
            v = vals.get((s, c))
            if v:
                line += f"{statistics.median(v):>{w}.1f}"
                drift.append(((max(v) - min(v)) / statistics.median(v), s, c, v))
            else:
                line += " " * (w - 1) + "-"
        print(line)
    drift.sort(reverse=True)
    print("\nlargest round-to-round spread (max-min)/median:")
    for d, s, c, v in drift[:8]:
        print(f"  {d:6.1%} {s} {c} {[round(x, 1) for x in v]}")


if __name__ == "__main__":
    main()
