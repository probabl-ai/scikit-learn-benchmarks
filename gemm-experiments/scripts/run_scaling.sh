#!/bin/bash
# Experiment 5: thread scaling, pinned (first T physical cores, then siblings).
# usage: [SWEEP_ARGS=...] scripts/run_scaling.sh MACHINE THREADS [OUTNAME] [extra variants...]
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
OUT=${3:-scaling}
shift 3 || shift $#
SHAPES=(
  128x128x128 256x256x256 512x512x512 1024x1024x1024 2048x2048x2048
  4096x4096x4096 8192x8192x8192
  syrk:990x463810:N syrk:1000x50000:N 100x100x800000:NT
  7x116203x100:NN 100x200000x100:NN gemv:126x36168:T gemv:126x36168:N
)
V=(--variant openblas --variant "mkl-close:mkl,OMP_PROC_BIND=close,OMP_PLACES=cores"
   --variant blis --variant "openblas-omp-close:openblas-omp,OMP_PROC_BIND=close,OMP_PLACES=cores")
for v in "$@"; do V+=(--variant "$v"); done
pixi run python scripts/sweep.py --out results/$MACHINE/$OUT.csv \
  --threads "$THREADS" "${V[@]}" --dtype d --rounds 2 --min-time 0.3 \
  --note "exp5 scaling, pinned compact" ${SWEEP_ARGS:---pin compact} --shapes "${SHAPES[@]}"
