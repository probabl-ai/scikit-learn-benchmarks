#!/bin/bash
# Experiment 16: from what size does threading a small syrk pay? PR #3 build
# (threads = min(T, n / switch_ratio)), f64, T from 1 to 86, pinned.
# usage: scripts/run_syrk_small.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
LLVM=$PWD/.pixi/envs/omp/lib
SHAPES=()
for n in 32 64 128 256 512; do
  for k in 1000 5000 20000 100000; do SHAPES+=(syrk:${n}x$k:N); done
done
pixi run python scripts/sweep.py --out results/$MACHINE/syrk_small.csv \
  --threads "$THREADS" \
  --variant "pthreads=builds/sr-syrk/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}" \
  --variant "omp-llvm=builds/sr-syrk-omp/lib/libopenblas.so.0,OMP_NUM_THREADS={T},LD_LIBRARY_PATH=$LLVM" \
  --dtype d --rounds 2 --min-time 0.2 --pin compact \
  --note "exp16 small syrk thread-count crossover" --shapes "${SHAPES[@]}"
