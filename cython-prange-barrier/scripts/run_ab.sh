#!/bin/bash
# Run under benchlock, on the host. usage: run_ab.sh PROFILE THREADS REPS [KEY=VAL ...]
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
P=$1; T=$2; R=$3; shift 3
eargs=(); for kv in "$@"; do eargs+=(--env "$kv"); done
CPUS=$(taskset -cp $$ | sed 's/.*: //')
echo "# $P T=$T cpus=$CPUS env=$* $(date -u +%H:%M)"
venv-m331/bin/python other_ab.py --builds m324,m331,mA,mB2 --threads $T --reps $((R * 2)) --out $P-other2.jsonl "${eargs[@]}"
venv-m331/bin/python bench_hgb.py --profile $P --threads $T --cpus "$CPUS" --cases year_prediction_msd,susy,L,kddcup09_churn --variants f331:pr,fB2:pr,f331:pr+fuse3,fB2:pr+fuse3,nb331:pr+fuse3 --n-runs 1 --reps $R "${eargs[@]}" --out $P-hgb2.jsonl
echo "# done $P $(date -u +%H:%M)"
