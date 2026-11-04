#!/bin/bash
# Run under benchlock, on the host. Repeat of main_fit_ab.sh with libomp, 3.3.1 vs Fix A vs Fix B, on more cases.
# usage: main_fit_libomp.sh PROFILE THREADS REPS LIBOMP
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
P=$1; T=$2; R=$3; L=$4
test -f "$L" || { echo "missing $L"; exit 1; }
CPUS=$(taskset -cp $$ | sed 's/.*: //')
echo "# $P T=$T cpus=$CPUS libomp=$L $(date -u +%H:%M)"
venv-m331/bin/python bench_hgb.py --profile "$P" --threads "$T" --cpus "$CPUS" \
  --cases year_prediction_msd,susy,L,kddcup09_churn --variants m331:main,mA:main,mB2:main \
  --n-runs 1 --reps "$R" --env "LD_PRELOAD=$L" --out "$P-mainfit.jsonl"
echo "# done $P $(date -u +%H:%M)"
