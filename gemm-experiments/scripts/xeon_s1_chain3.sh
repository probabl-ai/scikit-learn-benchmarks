#!/bin/bash
# Third s1 queue: build develop with all four patches, check, A/B.
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
lock "build OpenBLAS all patches, checks" bash -c '
  set -e
  pixi run build
  if [ ! -f builds/all/lib/libopenblas.so.0 ]; then
    rm -rf openblas-all && cp -a openblas openblas-all
    for p in splitk_gemm l1stride gemv_splitx syrk_nthreads; do
      pixi run python patches/apply_$p.py openblas-all
    done
    scripts/build_openblas.sh openblas-all builds/all
  fi
  scripts/check_patches.sh builds/all > results/xeon-s1/check_all.txt 2>&1'
lock "all patches A/B" scripts/run_all.sh xeon-s1 8,22,43,64,86
