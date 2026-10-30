"""Pivot a sweep CSV: median GFLOP/s per (shape, dtype, threads) and variant.

    python scripts/summarize.py results/xeon-s1/map.csv [--ref mkl] [--metric gflops_median]

Takes the best round (max of the per-round medians) to filter out noisy
rounds, then prints GFLOP/s per variant and the ratio of each variant to --ref.
"""

import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--ref", default="mkl")
    p.add_argument("--metric", default="gflops_median")
    p.add_argument("--dtype")
    p.add_argument("--threads")
    args = p.parse_args()
    df = pd.concat([pd.read_csv(c) for c in args.csv])
    df["shape"] = (df.m.astype(str) + "x" + df.n.astype(str) + "x"
                   + df.k.astype(str) + ":" + df.trans)
    if args.dtype:
        df = df[df.dtype == args.dtype]
    if args.threads:
        df = df[df.threads_requested.isin([int(t) for t in args.threads.split(",")])]
    keys = ["dtype", "threads_requested", "shape"]
    g = df.groupby(keys + ["variant"], sort=False)[args.metric].max().unstack("variant")
    variants = list(dict.fromkeys(df.variant))
    g = g[variants]
    out = g.round(1).astype(str)
    if args.ref in g:
        for v in variants:
            if v != args.ref:
                out[f"{v}/{args.ref}"] = (g[v] / g[args.ref]).round(2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 2000)
    pd.set_option("display.max_columns", 50)
    print(out.to_string())


if __name__ == "__main__":
    main()
