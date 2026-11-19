"""Throughput (calls completed in the window) against a base label: median
over rounds, and [min, max] of the per-round ratio."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
base = sys.argv[2] if len(sys.argv) > 2 else "base"
only = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None
metric = sys.argv[4] if len(sys.argv) > 4 else "reps"
df["shape"] = df.m.astype(str) + "x" + df.n.astype(str) + "x" + df.k.astype(str) + ":" + df.trans
df["round"] = df.groupby(["dtype", "T", "callers", "shape", "label"]).cumcount()
w = df.pivot_table(index=["dtype", "T", "callers", "shape", "round"], columns="label", values=metric, sort=False)
r = w.div(w[base], axis=0).drop(columns=base)
if only: r = r[only]
g = r.groupby(level=[0, 1, 2, 3], sort=False)
out = pd.DataFrame({c: g[c].median().round(2).astype(str) + " [" + g[c].min().round(2).astype(str) + "," + g[c].max().round(2).astype(str) + "]" for c in r.columns})
out.insert(0, "base calls", w[base].groupby(level=[0, 1, 2, 3], sort=False).median().round(0))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(out.to_string())
