"""Two BLAS processes at once on one socket: what does binding cost then?

OMP_PROC_BIND=close puts every process on the same first cores of its
affinity mask, so two bound processes can share cores while others idle. Each
case runs one shape in two processes started together (or one alone, as the
reference) and records each process's GFLOP/s.

    python scripts/collide.py --out results/xeon-s1/collide.csv --threads 43,86
"""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

from sweep import BIN, PRESETS, compact_cpus

BIND = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
# mode: (preset, extra env, number of processes, split the socket in halves)
MODES = {
    "openblas-alone": ("openblas", {}, 1, False),
    "openblas-x2": ("openblas", {}, 2, False),
    "mkl-alone": ("mkl", {}, 1, False),
    "mkl-x2": ("mkl", {}, 2, False),
    "mkl-close-alone": ("mkl", BIND, 1, False),
    "mkl-close-x2": ("mkl", BIND, 2, False),
    "mkl-close-x2-halves": ("mkl", BIND, 2, True),
}


def halves():
    cpus = compact_cpus(len(os.sched_getaffinity(0)))
    ncores = len({Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
                  .read_text().strip() for c in cpus})
    cores, sibs = cpus[:ncores], cpus[ncores:]
    h = ncores // 2
    return [cores[:h] + sibs[:h], cores[h:] + sibs[h:]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--threads", default="43,86")
    p.add_argument("--shapes", nargs="+",
                   default=["1024x1024x1024", "4096x4096x4096", "7x116203x100:NN"])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--min-time", default="2")
    args = p.parse_args()
    out = Path(args.out)
    fields = ["round", "threads", "shape", "mode", "proc", "gflops_best", "gflops_median",
              "reps", "affinity"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in range(args.rounds):
            for t in map(int, args.threads.split(",")):
                for shape in args.shapes:
                    for mode, (preset, extra, nproc, split) in MODES.items():
                        lib, env0 = PRESETS[preset]
                        env = dict(os.environ)
                        env.update({k: v.replace("{T}", str(t)) for k, v in env0.items()})
                        env.update(extra)
                        procs = []
                        for i in range(nproc):
                            cmd = [str(BIN), "--lib", str(lib), "--label", mode,
                                   "--min-time", args.min_time, "--threads", str(t), shape]
                            if split:
                                cmd = ["taskset", "-c", ",".join(map(str, halves()[i]))] + cmd
                            procs.append(subprocess.Popen(cmd, env=env, text=True,
                                                          stdout=subprocess.PIPE,
                                                          stderr=subprocess.PIPE))
                        for i, pr in enumerate(procs):
                            so, se = pr.communicate()
                            if pr.returncode:
                                raise SystemExit(se)
                            row = so.strip().split(",")
                            aff = [ln for ln in se.splitlines() if "affinity" in ln]
                            rec = dict(round=r, threads=t, shape=shape, mode=mode, proc=i,
                                       reps=row[7], gflops_best=row[10],
                                       gflops_median=row[11],
                                       affinity=aff[0] if aff else "")
                            w.writerow(rec)
                            print(f"r{r} T={t:>3} {shape:>18} {mode:>20} p{i} "
                                  f"{float(row[11]):8.1f} GF/s", flush=True)
                        f.flush()
                        time.sleep(0.5)


if __name__ == "__main__":
    main()
