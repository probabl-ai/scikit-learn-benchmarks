#!/bin/bash
# Experiment 17: PR #3 with the work-per-thread cap (sr-syrk2) vs develop
# and the first version (sr-syrk): small and large syrk, pthreads and OpenMP.
# usage: scripts/run_syrk_pr.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
LLVM=$PWD/.pixi/envs/omp/lib
SHAPES=(syrk:64x2000:N)
for n in 32 64 128 256 512; do
  for k in 1000 5000 20000 100000; do SHAPES+=(syrk:${n}x$k:N); done
done
SHAPES+=(syrk:990x463810:N syrk:1000x50000:N syrk:500x100000:N syrk:300x200000:N
         syrk:2000x100000:N syrk:990x2000:N syrk:200x2000:N)
V=()
for b in sr-base sr-syrk sr-syrk2; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
done
for b in sr-base sr-syrk2; do
  V+=(--variant "$b-omp-llvm=builds/$b-omp/lib/libopenblas.so.0,OMP_NUM_THREADS={T},LD_LIBRARY_PATH=$LLVM")
done
pixi run python scripts/sweep.py --out results/$MACHINE/syrk_pr.csv \
  --threads "$THREADS" "${V[@]}" --dtype d,s --rounds 3 --min-time 0.2 --pin compact \
  --note "exp17 syrk PR with work-per-thread cap" --shapes "${SHAPES[@]}"
