#!/bin/bash
# Experiment 10 (Xeon socket): high thread counts. OpenBLAS thread binding
# (NO_AFFINITY=0 build), the 128-thread cap (NUM_THREADS=256 build), and
# memory placement (first touch by one thread vs interleaved over the
# socket's SNC nodes).
# usage: scripts/run_scaling_hi.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
SHAPES=(1024x1024x1024 4096x4096x4096 8192x8192x8192 16384x16384x16384
        syrk:990x463810:N 100x200000x100:NN)
V=(--variant openblas --variant "mkl-close:mkl,OMP_PROC_BIND=close,OMP_PLACES=cores"
   --variant "openblas-omp-close:openblas-omp,OMP_PROC_BIND=close,OMP_PLACES=cores")
for b in t256 aff; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
done
NODES=$(numactl --show | sed -n 's/^membind: //p' | tr ' ' ',' | sed 's/,$//')
for mem in local interleave; do
  prefix=""
  [ $mem = interleave ] && prefix="numactl --interleave=$NODES"
  pixi run python scripts/sweep.py --out results/$MACHINE/scaling_hi_$mem.csv \
    --threads "$THREADS" "${V[@]}" --dtype d --rounds 2 --min-time 0.5 \
    --prefix-cmd "$prefix" --note "exp10 high-T scaling, memory=$mem" \
    --pin compact --shapes "${SHAPES[@]}"
done
