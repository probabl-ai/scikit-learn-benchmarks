#!/bin/bash
# Fourth s1 queue: all patches + syrk split-K, checks, A/B.
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
lock "build OpenBLAS all + syrk split-K, checks" bash -c '
  set -e
  pixi run build
  if [ ! -f builds/all2/lib/libopenblas.so.0 ]; then
    rm -rf openblas-all2 && cp -a openblas openblas-all2
    for p in splitk_gemm l1stride gemv_splitx syrk_nthreads splitk_syrk; do
      pixi run python patches/apply_$p.py openblas-all2
    done
    scripts/build_openblas.sh openblas-all2 builds/all2
  fi
  scripts/check_patches.sh builds/all2 > results/xeon-s1/check_all2.txt 2>&1'
BUILDS="base all all2" lock "all patches + syrk split-K A/B" scripts/run_all.sh xeon-s1 8,22,43,86 all2
