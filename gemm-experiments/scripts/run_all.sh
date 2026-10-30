#!/bin/bash
# Experiment 12: all four patches together (split-K v2, l1stride, gemv
# split-x, syrk thread count) vs develop, conda-forge OpenBLAS and bound MKL.
# usage: [SWEEP_ARGS=...] scripts/run_all.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
SHAPES=(
  100x100x1000000:NT 200x200x50000:TN 50x1000x100000:NT 1000x1000x100000:NT
  100x200000x100:NN 10x1000000x100:NN 7x116203x100:NN 50x100000x1000:NN
  256x100000x50:TN 256x256x256 1024x1024x1024 4096x4096x4096 2000x2000x200
  syrk:990x463810:N syrk:1000x50000:N syrk:300x200000:N syrk:2000x100000:N
  gemv:126x36168:N gemv:100x200000:N gemv:100x200000:T
)
V=(--variant openblas --variant "mkl-close:mkl,OMP_PROC_BIND=close,OMP_PLACES=cores")
for b in base all; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
done
pixi run python scripts/sweep.py --out results/$MACHINE/all.csv \
  --threads "$THREADS" "${V[@]}" --dtype d,s --rounds 3 --min-time 0.3 \
  --note "exp12 all patches A/B" ${SWEEP_ARGS:---pin compact} --shapes "${SHAPES[@]}"
