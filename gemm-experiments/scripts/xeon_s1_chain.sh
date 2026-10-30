#!/bin/bash
# Sequential queue of the remaining Xeon experiments on the s1 slot (s0 is
# used by other work). Each step takes the lock separately so other jobs can
# interleave between steps.
set -uo pipefail
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
lock "build OpenBLAS affinity + gemv split-x, checks" bash -c '
  set -e
  test -f builds/aff/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-aff builds/aff NUM_THREADS=256 NO_AFFINITY=0 BUILD_BFLOAT16=0
  test -f builds/gemv/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-gemv builds/gemv
  pixi run build
  scripts/check_patches.sh builds/combo builds/gemv > results/xeon-s1/check.txt 2>&1'
lock "combined patch A/B" scripts/run_combo.sh xeon-s1 8,22,43,86
lock "gemv split-x A/B" scripts/run_gemv.sh xeon-s1 1,2,4,8,22,43,86
lock "high-T scaling, affinity, memory placement" scripts/run_scaling_hi.sh xeon-s1 22,43,64,86,128,172
