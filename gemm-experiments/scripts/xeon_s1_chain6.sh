#!/bin/bash
# Sixth s1 queue: OpenMP builds for exp 15, then the cycle A/B.
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
lock "build OpenMP OpenBLAS develop/syrk PR" bash -c '
  set -e
  pixi run build
  for b in sr-base sr-syrk; do
    test -f builds/$b-omp/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-$b builds/$b-omp USE_OPENMP=1
  done
  ldd builds/sr-syrk-omp/lib/libopenblas.so.0 | grep -i omp'
lock "syrk thread-count switching A/B" scripts/run_syrk_cycle.sh xeon-s1 22,43,86
