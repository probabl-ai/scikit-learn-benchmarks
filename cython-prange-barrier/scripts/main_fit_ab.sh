#!/bin/bash
# Run under benchlock, on the host. Fit A/B of scikit-learn main built with each Cython, interleaved.
# usage: main_fit_ab.sh PROFILE THREADS REPS [KEY=VAL ...]
# The text after `:` in a variant is an env value the main source ignores (see parse_variant in bench_hgb.py).
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
P=$1; T=$2; R=$3; shift 3
eargs=(); for kv in "$@"; do eargs+=(--env "$kv"); done
CPUS=$(taskset -cp $$ | sed 's/.*: //')
echo "# $P T=$T cpus=$CPUS env=$* $(date -u +%H:%M)"
venv-m331/bin/python bench_hgb.py --profile "$P" --threads "$T" --cpus "$CPUS" \
  --cases year_prediction_msd --variants m324:main,m331:main,mA:main,mB2:main \
  --n-runs 1 --reps "$R" "${eargs[@]}" --out "$P-mainfit.jsonl"
echo "# done $P $(date -u +%H:%M)"
