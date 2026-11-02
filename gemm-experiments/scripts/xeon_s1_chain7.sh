#!/bin/bash
# Seventh s1 queue: PR #3 v2 (work-per-thread cap): build, checks, A/B, cycles.
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
git -C openblas-fork fetch -q origin syrk-fewer-threads
test -d openblas-sr-syrk2 || git -C openblas-fork worktree add -q --detach ../openblas-sr-syrk2 origin/syrk-fewer-threads
lock "build syrk PR v2, checks" bash -c '
  set -e
  pixi run build
  test -f builds/sr-syrk2/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sr-syrk2 builds/sr-syrk2
  test -f builds/sr-syrk2-omp/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sr-syrk2 builds/sr-syrk2-omp USE_OPENMP=1
  git -C openblas-sr-syrk2 log --oneline -1
  scripts/check_patches.sh builds/sr-syrk2 > results/xeon-s1/check_syrk2.txt 2>&1'
lock "syrk PR v2 A/B" scripts/run_syrk_pr.sh xeon-s1 8,22,43,86
BUILDS="sr-base sr-syrk2" BOUND=0 lock "syrk PR v2 cycles" scripts/run_syrk_cycle.sh xeon-s1 22,43,86 syrk2_cycle
