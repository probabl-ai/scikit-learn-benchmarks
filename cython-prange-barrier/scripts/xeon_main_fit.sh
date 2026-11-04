#!/bin/bash
# Run under benchlock on intel-xeon, one socket: 3.3.1 vs Fix B on main, year_prediction_msd.
# usage: xeon_main_fit.sh SLOT_THREADS. Thread count is explicit, as for socket slots.
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
T=$1
L=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-mkl/lib/libomp.so
test -f "$L" || { echo "missing $L"; exit 1; }
CPUS=$(taskset -cp $$ | sed 's/.*: //')
run() {  # NAME REPS [KEY=VAL ...]
  name=$1; R=$2; shift 2
  eargs=(); for kv in "$@"; do eargs+=(--env "$kv"); done
  echo "# $name T=$T cpus=$CPUS env=$* $(date -u +%H:%M)"
  venv-m331/bin/python bench_hgb.py --profile "$name" --threads "$T" --cpus "$CPUS" \
    --cases year_prediction_msd --variants m331:main,mB2:main \
    --n-runs 1 --reps "$R" "${eargs[@]}" --out xeon-probe.jsonl
}
run gomp_default 5
run omp_default 5 "LD_PRELOAD=$L"
run omp_blocktime0 3 "LD_PRELOAD=$L" KMP_BLOCKTIME=0
echo "# all done $(date -u +%H:%M)"
