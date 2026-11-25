"""Speed-up tables of exp 31 (each fork PR vs develop), one per PR.

Cell: median over rounds of the PR's GF/s divided by develop's, at the given
thread count of each machine. The last row is the range over every cell of the
PR's grid at that thread count.

    python scripts/prs_table.py PR MACHINE:THREADS[:LABEL] ...
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Rows shown for each PR: (dtype, m, n, k, trans) as gemmbench writes them,
# keyed by a short label.
ROWS = {
    "3": {
        "dsyrk 64 x 20000": ("d", "syrk:64x20000:N"),
        "dsyrk 128 x 20000": ("d", "syrk:128x20000:N"),
        "dsyrk 256 x 20000": ("d", "syrk:256x20000:N"),
        "dsyrk 512 x 20000": ("d", "syrk:512x20000:N"),
        "dsyrk 990 x 100000": ("d", "syrk:990x100000:N"),
        "ssyrk 128 x 20000": ("s", "syrk:128x20000:N"),
        "ssyrk 512 x 20000": ("s", "syrk:512x20000:N"),
    },
    "10": {
        "dsyrk 32 x 1e6": ("d", "syrk:32x1000000:N"),
        "dsyrk 64 x 2e5": ("d", "syrk:64x200000:N"),
        "ssyrk 64 x 2e5": ("s", "syrk:64x200000:N"),
        "dgemm NT 64x64, k = 2e5": ("d", "64x64x200000:NT"),
        "dgemm NT 128x128, k = 2e5": ("d", "128x128x200000:NT"),
        "dgemm TN 200x200, k = 5e4": ("d", "200x200x50000:TN"),
        "dgemm NT 10x200, k = 1e6": ("d", "10x200x1000000:NT"),
        "dgemm NT 1000x1000, k = 1e5 (no split)": ("d", "1000x1000x100000:NT"),
        "dgemm 2000^3 (no split)": ("d", "2000x2000x2000"),
    },
    "8": {
        "ddot 1e5": ("d", "dot:100000"),
        "daxpy 1e5": ("d", "axpy:100000"),
        "daxpy 1e6": ("d", "axpy:1000000"),
        "dgetrf 1000": ("d", "getrf:1000x1000"),
        "dgetrf 4000": ("d", "getrf:4000x4000"),
        "dgetrf 20000 x 500": ("d", "getrf:20000x500"),
        "dgesv 2000, 1 rhs": ("d", "gesv:2000x1"),
        "dgesv 1000, 10000 rhs": ("d", "gesv:1000x10000"),
        "dgesv 2000, 20000 rhs": ("d", "gesv:2000x20000"),
    },
    "11": {
        "dsymv 200": ("d", "symv:200"),
        "dsymv 500": ("d", "symv:500"),
        "dsymv 1000": ("d", "symv:1000"),
        "dsymv 4000": ("d", "symv:4000"),
    },
}


def spec_key(spec):
    """(trans, m, n, k) as gemmbench writes them in its CSV for a shape spec."""
    if spec.startswith("syrk:"):
        n, k, *t = spec[5:].replace(":", "x").split("x")
        return ("syrk" + (t[0] if t else "N"), n, n, k)
    if ":" in spec and not spec.split(":")[1].isalpha():
        name, dims = spec.split(":")
        d = dims.split("x")
        if len(d) == 1:
            return (name, d[0], d[0], "1")
        if name in ("gesv", "posv"):
            return (name, d[0], d[0], d[1])
        return (name, d[0], d[1], d[2] if len(d) > 2 else "1")
    dims, _, t = spec.partition(":")
    m, n, k = dims.split("x")
    return (t or "NN", m, n, k)


def speedups(pr, machine, threads):
    df = pd.read_csv(ROOT / "results" / machine / "prs" / f"pr{pr}.csv", dtype=str)
    df["gf"] = df["gflops_median"].astype(float)
    df = df[df["threads_requested"].astype(int) == threads]
    med = df.groupby(["dtype", "trans", "m", "n", "k", "variant"])["gf"].median()
    med = med.unstack("variant")
    return med[f"pr{pr}"] / med["develop"]


def fmt(x):
    return f"{x:.2f}x" if x < 2 else f"{x:.1f}x"


def main():
    pr, cols = sys.argv[1], sys.argv[2:]
    machines = []
    for c in cols:
        machine, threads, *label = c.split(":")
        machines.append((machine, int(threads), label[0] if label else f"{machine} {threads}T"))
    ratios = {label: speedups(pr, m, t) for m, t, label in machines}
    print("| call | " + " | ".join(label for *_, label in machines) + " |")
    print("| --- |" + " ---: |" * len(machines))
    for name, (dt, spec) in ROWS[pr].items():
        key = (dt, *spec_key(spec))
        cells = [fmt(r.get(key, float("nan"))) for r in ratios.values()]
        print(f"| {name} | " + " | ".join(cells) + " |")
    cells = [f"{fmt(r.min())[:-1]}-{fmt(r.max())}" for r in ratios.values()]
    n = min(len(r) for r in ratios.values())
    print(f"| all {n} cells | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
