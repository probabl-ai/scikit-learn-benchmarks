#!/bin/bash
# Second s1 queue: MKL runtime/binding experiments (exp11, collide), then what
# the first chain lost when the aff build failed its tests.
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
test -f vendor/intel-openmp/libiomp5.so || scripts/get_intel_openmp.sh
lock "build harness" pixi run build
lock "MKL OpenMP runtime/binding variants" scripts/run_mkl_runtime.sh xeon-s1 22,43,86,172
lock "two concurrent bound MKL processes" pixi run python scripts/collide.py --out results/xeon-s1/collide.csv --threads 43,86
lock "build OpenBLAS gemv split-x, checks" bash -c '
  set -e
  test -f builds/gemv/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-gemv builds/gemv
  scripts/check_patches.sh builds/combo builds/gemv > results/xeon-s1/check.txt 2>&1'
lock "gemv split-x A/B" scripts/run_gemv.sh xeon-s1 1,2,4,8,22,43,86
lock "high-T scaling, memory placement" scripts/run_scaling_hi.sh xeon-s1 22,43,64,86,128,172
