#!/bin/bash
# Experiment 9: gemv split-x patch vs develop, conda-forge and bound MKL.
# gemv:dxn:N is X.T @ r and gemv:dxn:T is X @ w for X (n, d) row-major.
# usage: [SWEEP_ARGS=...] scripts/run_gemv.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
SHAPES=(
  gemv:126x36168:N gemv:126x36168:T gemv:99x26215:N gemv:100x200000:N
  gemv:100x200000:T gemv:1000x40000:N gemv:1000x40000:T gemv:20x2000000:N
  gemv:20x2000000:T gemv:2199x40000:N gemv:4096x4096:N
)
V=(--variant openblas --variant "mkl-close:mkl,OMP_PROC_BIND=close,OMP_PLACES=cores")
for b in base gemv; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
done
pixi run python scripts/sweep.py --out results/$MACHINE/gemv.csv \
  --threads "$THREADS" "${V[@]}" --dtype d,s --rounds 3 --min-time 0.2 \
  --note "exp9 gemv split-x patch" ${SWEEP_ARGS:---pin compact} --shapes "${SHAPES[@]}"
