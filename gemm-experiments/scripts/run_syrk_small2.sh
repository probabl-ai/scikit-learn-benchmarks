#!/bin/bash
# Experiment 18: small syrk thread crossover in f32 and f64 (PR #3 v1 build,
# threads = min(T, n / switch_ratio), no work cap), pthreads, pinned.
# usage: scripts/run_syrk_small2.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
SHAPES=()
for n in 64 96 128 192 256; do
  for k in 1000 5000 20000 100000; do SHAPES+=(syrk:${n}x$k:N); done
done
pixi run python scripts/sweep.py --out results/$1/syrk_small2.csv --threads "$2" \
  --variant "v1=builds/sr-syrk/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}" \
  --dtype d,s --rounds 2 --min-time 0.2 --pin compact \
  --note "exp18 small syrk crossover f32/f64" --shapes "${SHAPES[@]}"
