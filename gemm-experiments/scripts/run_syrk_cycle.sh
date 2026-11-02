#!/bin/bash
# Experiment 15: does the syrk fewer-threads fix (fork PR #3) cost anything
# when consecutive syrk calls use different thread counts? Each process times
# every shape of a set alone, then the set called in a loop (--cycle 1).
# pthreads and OpenMP builds of develop and the PR; the OpenMP ones on LLVM
# libomp (what conda-forge ships as libgomp.so.1), LLVM bound, and GNU libgomp.
# usage: [BUILDS="sr-base sr-syrk"] [BOUND=1] scripts/run_syrk_cycle.sh MACHINE THREADS [OUTPREFIX]
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
OUT=${3:-syrk_cycle}
LLVM=$PWD/.pixi/envs/omp/lib
GNU=$PWD/vendor/gnu-gomp  # GNU libgomp.so.1 from conda-forge (libgomp, _openmp_mutex *_gnu)
BIND="OMP_PROC_BIND=close,OMP_PLACES=cores"
V=()
for b in ${BUILDS:-sr-base sr-syrk}; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
  V+=(--variant "$b-omp-llvm=builds/$b-omp/lib/libopenblas.so.0,OMP_NUM_THREADS={T},LD_LIBRARY_PATH=$LLVM")
  [ "${BOUND:-1}" = 1 ] && V+=(--variant "$b-omp-llvm-bound=builds/$b-omp/lib/libopenblas.so.0,OMP_NUM_THREADS={T},LD_LIBRARY_PATH=$LLVM,$BIND")
  V+=(--variant "$b-omp-gnu=builds/$b-omp/lib/libopenblas.so.0,OMP_NUM_THREADS={T},LD_LIBRARY_PATH=$GNU")
done
SETS=(
  "syrk:32x20000:N syrk:160x20000:N"
  "syrk:32x2000:N syrk:160x2000:N"
  "syrk:64x2000:N syrk:2000x2000:N"
  "syrk:200x2000:N syrk:2000x2000:N"
  "syrk:64x2000:N syrk:200x2000:N syrk:990x2000:N"
)
i=0
for set in "${SETS[@]}"; do
  pixi run python scripts/sweep.py --out results/$MACHINE/${OUT}_$i.csv \
    --threads "$THREADS" "${V[@]}" --dtype d --rounds 3 --min-time 0.3 \
    --harness-args "--cycle 1" --pin compact \
    --note "exp15 syrk thread-count changes between calls: $set" --shapes $set
  i=$((i + 1))
done
