"""Calls per second summed over the callers of a shape, against base."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
df["shape"] = df.spec.str.split(":").str[:4].str.join(":").str.replace(r"^(s:\d+:\d+):.*", r"\1", regex=True)
g = df.groupby(["T", "scenario", "window", "shape", "label", "round"], sort=False).calls_per_s.sum().unstack("label")
r = g.div(g["base"], axis=0).drop(columns="base")
gg = r.groupby(level=[0, 1, 2, 3], sort=True)
out = pd.DataFrame({c: gg[c].median().round(2).astype(str) + " [" + gg[c].min().round(2).astype(str) + "," + gg[c].max().round(2).astype(str) + "]" for c in r.columns})
out.insert(0, "base calls/s", g["base"].groupby(level=[0, 1, 2, 3], sort=True).median().round(1))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(out.to_string())
