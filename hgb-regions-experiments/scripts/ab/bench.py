"""Interleaved A/B of HGB variants on the hgb_scalability cases, at one thread count.

Run under benchlock with venv-main/bin/python and PYTHONPATH=~/scikit-learn-benchmarks
(the private venvs skip the shared sklearn-src editable install). Each (case, variant) is one sklbench runner
subprocess with --n-runs fits; variants are shuffled per case.
"""
import argparse, json, os, random, socket, statistics, subprocess, sys, tempfile, time
from pathlib import Path

B = Path.home() / "scikit-learn-benchmarks"
W = Path.home() / "work/hgb-regions"
VARIANTS = {
    "main": ("main", {}),
    "pr": ("exp", {"SKLEARN_HGB_GUARD": "pr"}),
    "abs": ("exp", {"SKLEARN_HGB_GUARD": "abs"}),
    "pr+fuse": ("exp", {"SKLEARN_HGB_GUARD": "pr", "SKLEARN_HGB_EXP": "fuse"}),
    "pr+fuse3": ("exp", {"SKLEARN_HGB_GUARD": "pr", "SKLEARN_HGB_EXP": "fuse3"}),
    "abs+fuse3": ("exp", {"SKLEARN_HGB_GUARD": "abs", "SKLEARN_HGB_EXP": "fuse3"}),
    "none+fuse3": ("exp", {"SKLEARN_HGB_GUARD": "none", "SKLEARN_HGB_EXP": "fuse3"}),
}

p = argparse.ArgumentParser()
p.add_argument("--profile", required=True)
p.add_argument("--threads", type=int, required=True)
p.add_argument("--variants", default=",".join(VARIANTS))
p.add_argument("--cases", default="", help="comma separated case names, default all")
p.add_argument("--env", action="append", default=[], help="KEY=VAL for every run")
p.add_argument("--n-runs", type=int, default=3)
p.add_argument("--reps", type=int, default=1)
p.add_argument("--out", required=True)
args = p.parse_args()

sys.path.insert(0, str(B / "configs"))
import hgb_scalability as cfg
cfg.get_n_cores_list = lambda **kw: [args.threads]
cases = cfg.generate_cases()
name = lambda c: c["data"].get("dataset") or c["metadata"]["name"]
if args.cases:
    keep = args.cases.split(",")
    cases = [c for c in cases if name(c) in keep]
extra_env = dict(kv.split("=", 1) for kv in args.env)
variants = args.variants.split(",")
rng = random.Random(0)
out = open(args.out, "a")
tmp = Path(tempfile.mkdtemp())
print(f"# {len(cases)} cases x {variants} profile={args.profile} T={args.threads} env={extra_env}", flush=True)
for rep in range(args.reps):
    for case in cases:
        order = variants[:]
        rng.shuffle(order)
        for v in order:
            build, venv_env = VARIANTS[v]
            case_file, res = tmp / "case.json", tmp / "res.jsonl"
            case_file.write_text(json.dumps(case))
            res.unlink(missing_ok=True)
            env = {**os.environ, **{k: str(x) for k, x in case["bench"]["env"].items()},
                   **venv_env, **extra_env, "PYTHONPATH": str(B)}
            for k in ("SKLEARN_HGB_GUARD", "SKLEARN_HGB_EXP"):
                if k not in venv_env:
                    env.pop(k, None)
            cmd = [str(W / f"venv-{build}/bin/python"), "-m", "sklbench.runners.estimator",
                   "--case-file", str(case_file), "--n-runs", str(args.n_runs), "--output-jsonl", str(res)]
            aff = case["bench"].get("cpu_affinity")
            if aff:
                cmd = ["taskset", "-c", ",".join(map(str, aff))] + cmd
            t0 = time.time()
            try:
                r = subprocess.run(cmd, env=env, cwd=B, capture_output=True, text=True, timeout=1200)
                rc, err = r.returncode, r.stderr[-2000:]
            except subprocess.TimeoutExpired:
                rc, err = "timeout", ""
            rows = [json.loads(l) for l in res.read_text().splitlines()] if res.exists() else []
            fits = [row["time_ms"]["fit"] / 1000 for row in rows]
            rec = {"host": socket.gethostname(), "profile": args.profile, "threads": args.threads,
                   "case": name(case), "variant": v, "rep": rep, "rc": rc, "fit_s": fits,
                   "rows": rows, "wall_s": time.time() - t0}
            if rc != 0:
                rec["stderr"] = err
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            med = f"{statistics.median(fits):.3f}" if fits else "NA"
            print(f"{args.profile} rep={rep} {name(case):24s} {v:10s} rc={rc} fit_med={med}", flush=True)
print("# done", flush=True)
