#!/bin/bash
# Experiment 6 (intel-gamer-gpu): P cores (CPUs 0-7) vs E cores (8-23).
# Same sweep restricted to different CPU sets with taskset.
# usage: scripts/run_hybrid.sh MACHINE
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
SHAPES=(512x512x512 1024x1024x1024 4096x4096x4096 100x200000x100:NN 2000x2000x200)
V=(--variant openblas --variant mkl --variant blis
   --variant "l1=builds/l1/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
run() {  # NAME CPUS THREADS
  pixi run python scripts/sweep.py --out results/$MACHINE/hybrid_$1.csv \
    --threads "$3" "${V[@]}" --dtype d,s --rounds 2 --min-time 0.3 \
    --prefix-cmd "taskset -c $2" --note "exp6 hybrid: $1 = CPUs $2" \
    --shapes "${SHAPES[@]}"
}
run p1 0 1
run e1 8 1
run p8 0-7 8
run e16 8-23 16
run all24 0-23 24
run all24-t8 0-23 8
