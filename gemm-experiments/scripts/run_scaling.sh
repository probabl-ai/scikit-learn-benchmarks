#!/bin/bash
# Experiment 2: thread scaling of GEMM and the sklearn-relevant routines.
# usage: scripts/run_scaling.sh MACHINE THREADS [extra sweep.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
shift 2
SHAPES=(
  2048x2048x2048 4096x4096x4096 8192x8192x8192
  syrk:990x463810:N syrk:1000x50000:N 100x100x800000:NT
  7x116203x100:NN 100x200000x100:NN gemv:126x36168:T gemv:126x36168:N
)
pixi run python scripts/sweep.py --out results/$MACHINE/scaling.csv \
  --threads "$THREADS" --variant openblas --variant mkl --variant blis \
  --dtype d --rounds 2 --min-time 0.3 \
  --note "exp2 scaling: default library settings, no pinning" "$@" ${SWEEP_ARGS:-} --shapes "${SHAPES[@]}"
