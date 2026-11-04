#!/bin/bash
# Exp 16: split_indices serial threshold sweep with fuse3. Under benchlock.
# usage: run_exp16.sh THREADS PROFILE CASES [KEY=VAL ...]
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
T=$1; P=$2; C=$3; shift 3
eargs=(); for kv in "$@"; do eargs+=(--env "$kv"); done
V=pr+fuse3,none+fuse3
for X in 10000 30000 100000 300000 1000000 2000000000; do V=$V,abs+fuse3@$X; done
venv-main/bin/python bench.py --profile $P --threads $T --cases $C --variants $V --n-runs 1 --reps ${REPS:-2} --phases "${eargs[@]}" --out $P.jsonl
