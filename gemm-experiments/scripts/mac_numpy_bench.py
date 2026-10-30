"""Time scikit-learn-like matmuls through numpy's PyPI wheels on macOS.

numpy's macosx_14_0 wheel calls Accelerate (cblas_?gemm$NEWLAPACK$ILP64, and
?syrk for X.T @ X); the macosx_11_0 wheel bundles scipy-openblas64. Each
(venv, env) pair runs in its own process, interleaved over rounds.

    python scripts/mac_numpy_bench.py --out results/mac-m2/x.csv \
        --venv accel=venvs/accel --venv openblas=venvs/openblas \
        --env 'default=' --env 't1=OPENBLAS_NUM_THREADS=1,VECLIB_MAXIMUM_THREADS=1'
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

CASES = [
    # name, shapes, dtype; flops computed in the worker
    ("XtX_syrk", "1000000x100"),
    ("XtY_gemm", "1000000x100"),
    ("X@W10", "1000000x100,100x10"),
    ("X@Ct8", "1000000x100,8x100"),
    ("X@Ct256", "100000x100,256x100"),
    ("X@Q20", "100000x1000,1000x20"),
    ("mlp_200x784@784x100", "200x784,784x100"),
    ("sq512", "512x512,512x512"),
    ("sq2048", "2048x2048,2048x2048"),
]


def worker(dtype, min_time):
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for name, shapes in CASES:
        dims = [tuple(int(v) for v in s.split("x")) for s in shapes.split(",")]
        X = rng.standard_normal(dims[0]).astype(dtype)
        if name == "XtX_syrk":
            f = lambda: X.T @ X  # noqa: E731
            flops = X.shape[0] * X.shape[1] * (X.shape[1] + 1)
        elif name == "XtY_gemm":
            Y = X.copy()
            f = lambda: X.T @ Y  # noqa: E731
            flops = 2 * X.shape[0] * X.shape[1] ** 2
        elif name.startswith("X@Ct"):
            C = rng.standard_normal(dims[1]).astype(dtype)
            f = lambda: X @ C.T  # noqa: E731
            flops = 2 * X.shape[0] * X.shape[1] * C.shape[0]
        else:
            W = rng.standard_normal(dims[1]).astype(dtype)
            f = lambda: X @ W  # noqa: E731
            flops = 2 * X.shape[0] * X.shape[1] * W.shape[1]
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.05:
            f()
        times = []
        t0 = time.perf_counter()
        while len(times) < 3 or time.perf_counter() - t0 < min_time:
            a = time.perf_counter()
            f()
            times.append(time.perf_counter() - a)
        times.sort()
        rows.append(dict(case=name, dtype=dtype, reps=len(times), best_s=times[0],
                         median_s=times[len(times) // 2],
                         gflops_best=flops / times[0] * 1e-9,
                         gflops_median=flops / times[len(times) // 2] * 1e-9))
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    print(json.dumps(dict(numpy=np.__version__, blas=blas, rows=rows)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--out")
    p.add_argument("--venv", action="append", default=[])
    p.add_argument("--env", action="append", default=["default="])
    p.add_argument("--dtype", default="float64,float32")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--min-time", type=float, default=0.5)
    args = p.parse_args()
    if args.worker:
        return worker(args.dtype, args.min_time)
    fields = ["round", "venv", "env", "numpy", "blas", "case", "dtype", "reps",
              "best_s", "median_s", "gflops_best", "gflops_median"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in range(args.rounds):
            for envspec in args.env:
                ename, _, kvs = envspec.partition("=")
                extra = dict(kv.split("=", 1) for kv in kvs.split(",") if kv)
                for vspec in args.venv:
                    vname, vpath = vspec.split("=", 1)
                    for dtype in args.dtype.split(","):
                        env = {**os.environ, **extra}
                        cmd = [f"{vpath}/bin/python", __file__, "--worker",
                               "--dtype", dtype, "--min-time", str(args.min_time)]
                        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
                        if res.returncode:
                            sys.exit(res.stderr)
                        d = json.loads(res.stdout)
                        for row in d["rows"]:
                            w.writerow(dict(round=r, venv=vname, env=ename,
                                            numpy=d["numpy"], blas=d["blas"], **row))
                            print(f"r{r} {ename:>8} {vname:>9} {dtype} {row['case']:>22} "
                                  f"{row['gflops_median']:8.1f} GF/s", flush=True)
                        f.flush()


if __name__ == "__main__":
    main()
