#!/bin/bash
# PR build (already installed in sklearn-dev): does keeping find_split/split_indices parallel
# (SKLEARN_OMP_WAIT_POLICY=active, runtime still passive) bring hist_time back to main's level?
# Runs through `pixi run` so the libomp env really gets libomp.
cd ~/scikit-learn-benchmarks
D=~/work/gamer-run/msd
OUT=$D/msd2.jsonl
echo "sklearn-src=$(git -C sklearn-src log --oneline -1)"
one() {  # env label omp hgb_n extra-env...
  local env=$1 label=$2 omp=$3 n=$4; shift 4
  rm -f $D/tmp.jsonl
  env OMP_NUM_THREADS=$omp HGB_N_THREADS=$n "$@" pixi run --frozen -e $env python $D/patched_runner.py --case-file $D/case.json --n-runs 3 --output-jsonl $D/tmp.jsonl 2> $D/tmp.err
  local rc=$?
  pixi run --frozen -e $env python -c "
import json, statistics as st
rows = [json.loads(l) for l in open('$D/tmp.jsonl')]
a = [r['attributes'] for r in rows]
med = lambda k: st.median(x[k] for x in a)
print(json.dumps(dict(env='$env', label='$label', omp=$omp, n='$n', rc=$rc, tree_n_threads=a[0]['tree_n_threads'],
  fit=[round(r['time_ms']['fit']/1000, 2) for r in rows],
  **{k: round(med(k), 2) for k in ['hist_time', 'find_split_time', 'apply_split_time', 'grower_init_time']})))
" | tee -a $OUT || { echo "FAILED $label rc=$rc"; tail -n 5 $D/tmp.err; }
}
for e in sklearn-dev sklearn-dev-libomp; do
  SHOW_POOLS=1 pixi run --frozen -e $e python $D/patched_runner.py --case-file $D/case.json --n-runs 1 --output-jsonl /dev/null 2>&1 | grep POOL
done
for rep in 1 2 3; do
  for e in sklearn-dev sklearn-dev-libomp; do
    one $e pr_default 24 ""
    one $e pr_policy_active 24 "" SKLEARN_OMP_WAIT_POLICY=active
    one $e pr_policy_passive 24 "" SKLEARN_OMP_WAIT_POLICY=passive
    one $e pr_n18 24 18
  done
done
echo "== msd2 done $(date -Is)"
