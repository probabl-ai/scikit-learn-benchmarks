#!/bin/bash
# Tenth s1 queue: thread-switching cycles (exp 15) on PR #3 v4.
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
lock "build syrk PR v4 OpenMP" bash -c '
  test -f builds/sr-syrk4-omp/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sr-syrk4 builds/sr-syrk4-omp USE_OPENMP=1'
BUILDS="sr-base sr-syrk4" BOUND=0 lock "syrk PR v4 cycles" scripts/run_syrk_cycle.sh xeon-s1 22,43,86 syrk4_cycle
