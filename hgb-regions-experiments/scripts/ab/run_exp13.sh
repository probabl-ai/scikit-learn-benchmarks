#!/bin/bash
# Exp 13: start barrier of prange. usage: run_exp13.sh THREADS PROFILE [KEY=VAL ...]
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
T=$1; P=$2; shift 2
eargs=(); for kv in "$@"; do eargs+=(--env "$kv"); done
V=${V:-pr,nb:pr,pr+fuse3,nb:pr+fuse3}
C=${C:-year_prediction_msd,susy,L,L-leaf255,kddcup09_churn}
venv-main/bin/python bench.py --profile $P --threads $T --cases $C --variants $V --n-runs 1 --reps ${REPS:-3} --phases "${eargs[@]}" --out $P.jsonl
