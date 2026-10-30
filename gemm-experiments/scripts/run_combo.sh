#!/bin/bash
# Experiment 7: combined patch (split-K v2 + l1stride) vs develop, conda-forge
# OpenBLAS and thread-bound MKL; 3 rounds to catch unstable variants.
# usage: [SWEEP_ARGS=...] scripts/run_combo.sh MACHINE THREADS [OUTNAME]
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
OUT=${3:-combo}
SHAPES=(
  100x100x1000000:NT 200x200x50000:TN 50x1000x100000:NT 1000x1000x100000:NT
  100x200000x100:NN 10x1000000x100:NN 7x116203x100:NN 50x100000x1000:NN
  256x100000x50:TN 256x256x256 1024x1024x1024 4096x4096x4096 2000x2000x200
)
V=(--variant openblas --variant "mkl-close:mkl,OMP_PROC_BIND=close,OMP_PLACES=cores")
for b in base l1 combo; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
done
pixi run python scripts/sweep.py --out results/$MACHINE/$OUT.csv \
  --threads "$THREADS" "${V[@]}" --dtype d,s --rounds 3 --min-time 0.3 \
  --note "exp7 combined patch A/B" ${SWEEP_ARGS:---pin compact} --shapes "${SHAPES[@]}"
