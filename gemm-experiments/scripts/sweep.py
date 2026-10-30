"""Run gemmbench over BLAS variants x thread counts x shapes, interleaved.

Each (round, threads, variant) is one gemmbench process timing every shape, so
the variants of an A/B comparison alternate within each round.

    python scripts/sweep.py --out results/xeon-s1/foo.csv --threads 1,8,86 \
        --variant openblas --variant mkl --variant mine=/path/libopenblas.so \
        --shapes 4096x4096x4096 1000000x100x100:TN

A variant is a preset name (see PRESETS) or NAME=LIBPATH[,KEY=VAL...]; the
KEY=VAL pairs are extra environment variables, and {T} in a value is replaced
by the thread count.
"""

import argparse
import csv
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENVS = ROOT / ".pixi" / "envs"
BIN = ROOT / "build" / "gemmbench"
SO = ".dylib" if sys.platform == "darwin" else ".so"

PRESETS = {
    "openblas": (ENVS / "default/lib/libopenblas.so.0", {"OPENBLAS_NUM_THREADS": "{T}"}),
    "openblas-omp": (ENVS / "omp/lib/libopenblas.so.0", {"OMP_NUM_THREADS": "{T}"}),
    "mkl": (ENVS / "default/lib/libmkl_rt.so", {"MKL_NUM_THREADS": "{T}"}),
    "blis": (ENVS / f"default/lib/libblis{SO}", {"BLIS_NUM_THREADS": "{T}"}),
    "accelerate": (
        Path("/System/Library/Frameworks/Accelerate.framework/Accelerate"),
        {"VECLIB_MAXIMUM_THREADS": "{T}"},
    ),
}
if sys.platform == "darwin":
    PRESETS["openblas"] = (ENVS / "default/lib/libopenblas.0.dylib", PRESETS["openblas"][1])
    PRESETS["openblas-omp"] = (ENVS / "omp/lib/libopenblas.0.dylib", PRESETS["openblas-omp"][1])


def parse_variant(spec):
    if "=" not in spec.split(",")[0]:
        name, *extra = spec.split(",")
        path, env = PRESETS[name]
        env = dict(env)
    else:
        head, *extra = spec.split(",")
        name, path = head.split("=", 1)
        env = {}
        path = Path(path)
    for kv in extra:
        k, v = kv.split("=", 1)
        env[k] = v
    return name, str(path), env


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout
    except Exception as e:  # noqa: BLE001
        return repr(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--threads", default="1")
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--shapes", nargs="+", required=True)
    p.add_argument("--dtype", default="d")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--min-time", default="0.3")
    p.add_argument("--init", default="serial")
    p.add_argument("--prefix-cmd", default="", help="e.g. 'numactl --interleave=all'")
    p.add_argument("--note", default="")
    args = p.parse_args()

    variants = [parse_variant(v) for v in args.variant]
    threads = [int(t) for t in args.threads.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "argv": sys.argv,
        "note": args.note,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variants": {n: {"lib": path, "env": env} for n, path, env in variants},
        "lscpu": run_cmd("lscpu || sysctl -a | grep -E 'machdep.cpu|hw.perflevel|hw.memsize'"),
        "affinity": run_cmd("numactl --show || true"),
        "loadavg_start": os.getloadavg(),
        "lib_headers": {},
    }
    fields = ["round", "variant", "dtype", "m", "n", "k", "trans", "threads_requested",
              "threads", "reps", "best_s", "median_s", "gflops_best", "gflops_median"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in range(args.rounds):
            for t in threads:
                for name, path, extra in variants:
                    env = dict(os.environ)
                    env.update({k: v.replace("{T}", str(t)) for k, v in extra.items()})
                    for dtype in args.dtype.split(","):
                        cmd = args.prefix_cmd.split() + [
                            str(BIN), "--lib", path, "--label", name, "--dtype", dtype,
                            "--min-time", args.min_time, "--init", args.init,
                            "--threads", str(t), *args.shapes,
                        ]
                        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
                        meta["lib_headers"].setdefault(name, res.stderr.strip())
                        if res.returncode:
                            print(res.stderr, file=sys.stderr)
                            raise SystemExit(res.returncode)
                        for row in csv.reader(io.StringIO(res.stdout)):
                            (label, dt, m, n, k, tr, thr, reps, best, med, gb, gm) = row
                            rec = dict(round=r, variant=label, dtype=dt, m=m, n=n, k=k,
                                       trans=tr, threads_requested=t, threads=thr,
                                       reps=reps, best_s=best, median_s=med,
                                       gflops_best=gb, gflops_median=gm)
                            w.writerow(rec)
                            print(f"r{r} T={t:>3} {name:>14} {dt} {m}x{n}x{k}:{tr} "
                                  f"{float(gm):9.1f} GF/s (best {float(gb):9.1f})",
                                  flush=True)
                    f.flush()
    meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["loadavg_end"] = os.getloadavg()
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
