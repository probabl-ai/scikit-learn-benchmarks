#!/bin/bash
# year_prediction_msd HGB fit: thread count of the HGB team vs OMP_NUM_THREADS, default (passive) wait.
# usage: run_msd.sh ENV TAG
cd ~/scikit-learn-benchmarks
ENV=$1; TAG=$2
PY=.pixi/envs/$ENV/bin/python
D=~/work/gamer-run/msd
OUT=$D/$TAG.jsonl
echo "env=$ENV sklearn-src=$(git -C sklearn-src log --oneline -1)"
SHOW_POOLS=1 $PY $D/patched_runner.py --case-file $D/case.json --n-runs 1 --output-jsonl /dev/null 2>&1 | grep POOL
one() {  # label omp hgb_n extra-env...
  local label=$1 omp=$2 n=$3; shift 3
  rm -f $D/tmp.jsonl
  env OMP_NUM_THREADS=$omp HGB_N_THREADS=$n "$@" $PY $D/patched_runner.py --case-file $D/case.json --n-runs 3 --output-jsonl $D/tmp.jsonl 2> $D/tmp.err
  local rc=$?
  $PY -c "
import json, statistics as st
rows = [json.loads(l) for l in open('$D/tmp.jsonl')]
a = [r['attributes'] for r in rows]
med = lambda k: st.median(x[k] for x in a)
print(json.dumps(dict(label='$label', omp=$omp, n='$n', rc=$rc, tree_n_threads=a[0]['tree_n_threads'],
  fit=[round(r['time_ms']['fit']/1000, 2) for r in rows],
  **{k: round(med(k), 2) for k in ['hist_time', 'find_split_time', 'apply_split_time', 'grower_init_time']})))
" | tee -a $OUT || { echo "FAILED $label rc=$rc"; tail -n 5 $D/tmp.err; }
}
for rep in 1 2; do
  one omp24_n24 24 ""
  one omp24_n23 24 23
  one omp23_nA 23 ""
  one omp24_n22 24 22
  one omp24_n20 24 20
  one omp24_n16 24 16
  one omp16_nA 16 ""
  one omp24_n23_spin 24 23 GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms
done
echo "== msd done $(date -Is)"
