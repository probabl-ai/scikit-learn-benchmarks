#!/bin/bash
# Run under benchlock, on the host. usage: run_ab.sh PROFILE THREADS REPS [KEY=VAL ...]
cd ~/work/cython-prange; export PYTHONPATH=~/scikit-learn-benchmarks
P=$1; T=$2; R=$3; shift 3; eargs=(); for kv in "$@"; do eargs+=(--env "$kv"); done
CPUS=$(taskset -cp $$ | sed 's/.*: //')
PY=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev/bin/python
echo "# $P T=$T cpus=$CPUS env=$* $(date -u +%H:%M)"
venv-m331/bin/python other_ab.py --builds m331,mA,mB --threads $T --reps $R --out $P-other.jsonl "${eargs[@]}"
venv-m331/bin/python bench_hgb.py --profile $P --threads $T --cpus "$CPUS" --cases year_prediction_msd,susy,L,kddcup09_churn --variants f331:pr,fB:pr,f331:pr+fuse3,fB:pr+fuse3,nb331:pr+fuse3 --n-runs 1 --reps $R "${eargs[@]}" --out $P-hgb.jsonl
echo "# done $P $(date -u +%H:%M)"
