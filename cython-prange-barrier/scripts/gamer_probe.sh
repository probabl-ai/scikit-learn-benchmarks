#!/bin/bash
# Run under benchlock on intel-gamer-gpu: why is Fix B slower on main with libomp?
# Each config is an interleaved 3.3.1 vs Fix B A/B on year_prediction_msd. P-cores are 0-7, E-cores 8-23.
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
L=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
test -f "$L" || { echo "missing $L"; exit 1; }
run() {  # NAME THREADS CPUS [KEY=VAL ...]
  name=$1; T=$2; C=$3; shift 3
  eargs=(--env "LD_PRELOAD=$L"); for kv in "$@"; do eargs+=(--env "$kv"); done
  echo "# $name T=$T cpus=$C env=$* $(date -u +%H:%M)"
  venv-m331/bin/python bench_hgb.py --profile "$name" --threads "$T" --cpus "$C" \
    --cases year_prediction_msd --variants m331:main,mB2:main \
    --n-runs 1 --reps 5 "${eargs[@]}" --out gamer-probe.jsonl
}
run base 24 0-23
run spin 24 0-23 KMP_BLOCKTIME=200
run bound 24 0-23 OMP_PROC_BIND=true OMP_PLACES=cores
run p_only 8 0-7
run e_only 16 8-23
run base_again 24 0-23
echo "# all done $(date -u +%H:%M)"
