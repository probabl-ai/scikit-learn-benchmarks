#!/bin/bash
# Exp 18: noise level of the A/B, with the clean branch. 5 processes of 1 fit per case and
# variant, all 15 cases, interleaved. Under benchlock.
# usage: run_exp18.sh THREADS "PROFILE|VARIANTS|KEY=VAL KEY=VAL" ...
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
bash build_branch.sh clean hgb/merge-regions || exit 1
for v in main exp clean; do
  (cd /tmp && OMP_WAIT_POLICY=passive ~/work/hgb-regions/venv-$v/bin/python ~/work/hgb-regions/identity.py ~/work/hgb-regions/identity-$v.json)
done
venv-main/bin/python - <<PY
import json
a = {v: json.load(open(f"identity-{v}.json")) for v in ("main", "exp", "clean")}
for k in a["main"]:
    print(k, "clean==pr", a["clean"][k] == a["exp"][k], "pr==main", a["exp"][k] == a["main"][k])
PY
T=$1; shift
for spec in "$@"; do
  IFS='|' read -r prof variants envs <<< "$spec"
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  venv-main/bin/python bench.py --profile $prof --threads $T --variants "$variants" --n-runs 1 --reps ${REPS:-5} --phases "${eargs[@]}" --out $prof.jsonl
done
echo "### all done"
