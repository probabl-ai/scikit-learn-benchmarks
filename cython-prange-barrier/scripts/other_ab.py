"""Interleaved A/B of other_case.py across builds. Run under benchlock.

usage: other_ab.py --builds m331,mA,mB --threads T --reps R --out F [--env K=V ...] [--cases a,b]
"""
import argparse, json, os, random, subprocess
from pathlib import Path

W = Path.home() / "work/cython-prange"
p = argparse.ArgumentParser()
p.add_argument("--builds", required=True)
p.add_argument("--threads", type=int, required=True)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--out", required=True)
p.add_argument("--env", action="append", default=[])
p.add_argument("--cases", default="kmeans_lloyd,kmeans_elkan,kmeans_small,argkmin,argkmin_small,radius,tsne,hgb_predict")
args = p.parse_args()
env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       **dict(kv.split("=", 1) for kv in args.env)}
rng = random.Random(0)
builds = args.builds.split(",")
with open(args.out, "a") as out:
    for rep in range(args.reps):
        for case in args.cases.split(","):
            order = builds[:]
            rng.shuffle(order)
            for b in order:
                r = subprocess.run([str(W / f"venv-{b}/bin/python"), str(W / "other_case.py"), case, str(args.threads)],
                                   env=env, cwd=W, capture_output=True, text=True, timeout=1200)
                rec = {"build": b, "rep": rep, "rc": r.returncode, "env": args.env}
                try:
                    rec.update(json.loads(r.stdout.strip().splitlines()[-1]))
                except Exception:
                    rec.update({"case": case, "stderr": r.stderr[-1500:]})
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(b, case, rec.get("time_s"), rec.get("check"), rec["rc"], flush=True)
