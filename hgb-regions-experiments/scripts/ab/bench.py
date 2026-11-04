"""Interleaved A/B of HGB variants on the hgb_scalability cases, at one thread count.

Run under benchlock with venv-main/bin/python and PYTHONPATH=~/scikit-learn-benchmarks
(the private venvs skip the shared sklearn-src editable install). Each (case, variant) is one sklbench runner
subprocess with --n-runs fits; variants are shuffled per case.
"""
import argparse, json, os, random, socket, statistics, subprocess, sys, tempfile, time
from pathlib import Path

B = Path(os.environ.get("SKLBENCH_REPO", Path.home() / "scikit-learn-benchmarks"))
W = Path.home() / "work/hgb-regions"
DEFAULT_VARIANTS = "main,pr,abs,pr+fuse,pr+fuse3,abs+fuse3,none+fuse3"


def parse_variant(name):
    """`main`, or `[BUILD:]GUARD[+FUSE][@ABS_SAMPLES]`, e.g. `abs+fuse3@50000`, `nb:pr+fuse3`."""
    if name == "main":
        return "main", {}
    build, _, rest = name.rpartition(":")
    if rest == "main":  # a build without the experiment switches
        return build, {}
    rest, _, samples = rest.partition("@")
    guard, _, fuse = rest.partition("+")
    env = {"SKLEARN_HGB_GUARD": guard}
    if fuse:
        env["SKLEARN_HGB_EXP"] = fuse
    if samples:
        env["SKLEARN_HGB_ABS_SAMPLES"] = samples
    return build or "exp", env



p = argparse.ArgumentParser()
p.add_argument("--profile", required=True)
p.add_argument("--threads", type=int, required=True)
p.add_argument("--variants", default=DEFAULT_VARIANTS)
p.add_argument("--cases", default="", help="comma separated case names, default all")
p.add_argument("--env", action="append", default=[], help="KEY=VAL for every run")
p.add_argument("--n-runs", type=int, default=3)
p.add_argument("--reps", type=int, default=1)
p.add_argument("--phases", action="store_true", help="also record the grower phase times")
p.add_argument("--cpus", default="", help="taskset CPU list, replaces the case's cpu_affinity")
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
if "LD_PRELOAD" in extra_env and not Path(extra_env["LD_PRELOAD"]).is_file():
    # ld.so only warns on stderr, and the run silently uses the default runtime.
    sys.exit(f"LD_PRELOAD not found: {extra_env['LD_PRELOAD']}")
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
            build, venv_env = parse_variant(v)
            case_file, res = tmp / "case.json", tmp / "res.jsonl"
            case_file.write_text(json.dumps(case))
            res.unlink(missing_ok=True)
            env = {**os.environ, **{k: str(x) for k, x in case["bench"]["env"].items()},
                   **venv_env, **extra_env, "PYTHONPATH": str(B)}
            for k in ("SKLEARN_HGB_GUARD", "SKLEARN_HGB_EXP", "SKLEARN_HGB_ABS_SAMPLES"):
                if k not in venv_env:
                    env.pop(k, None)
            phase_file = tmp / "phases.json"
            phase_file.unlink(missing_ok=True)
            runner = ["-m", "sklbench.runners.estimator"]
            if args.phases:
                env["HGB_PHASE_OUT"] = str(phase_file)
                runner = [str(W / "phase_runner.py")]
            cmd = [str(W / f"venv-{build}/bin/python"), *runner,
                   "--case-file", str(case_file), "--n-runs", str(args.n_runs), "--output-jsonl", str(res)]
            aff = case["bench"].get("cpu_affinity")
            if args.cpus:
                cmd = ["taskset", "-c", args.cpus] + cmd
            elif aff:
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
            if phase_file.exists():
                rec["phases"] = json.loads(phase_file.read_text())
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            med = f"{statistics.median(fits):.3f}" if fits else "NA"
            print(f"{args.profile} rep={rep} {name(case):24s} {v:10s} rc={rc} fit_med={med}", flush=True)
print("# done", flush=True)
