#!/bin/bash
# Rerun the segfaulting case (sklearn-dev-libomp @ main b8de23f, forced active wait).
cd ~/scikit-learn-benchmarks
PY=.pixi/envs/sklearn-dev-libomp/bin/python
D=~/work/gamer-run/segv
echo "sklearn-src at $(git -C sklearn-src log --oneline -1)"
ulimit -c unlimited
fails=0
run() {  # $1 = threads, $2 = repetitions
  for i in $(seq 1 $2); do
    sed "s/\"OMP_NUM_THREADS\": \"8\"/\"OMP_NUM_THREADS\": \"$1\"/" $D/case.json > $D/case_$1.json
    PYTHONFAULTHANDLER=1 GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms OMP_NUM_THREADS=$1 \
      timeout 120 $PY -m sklbench.runners.estimator --case-file $D/case_$1.json --n-runs 5 \
      --output-jsonl $D/out.jsonl > $D/last.out 2> $D/last.err
    rc=$?
    if [ $rc -ne 0 ]; then fails=$((fails+1)); echo "T=$1 run $i rc=$rc $(date -Is)"; cp $D/last.err $D/fail_T$1_$i.err; tail -n 30 $D/last.err; fi
  done
  echo "T=$1 done ($2 runs), failures so far: $fails"
}
run 8 50
for T in 2 4 16 24; do run $T 15; done
echo "== repro done fails=$fails $(date -Is)"
