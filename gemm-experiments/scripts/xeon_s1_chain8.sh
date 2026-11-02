#!/bin/bash
# Eighth s1 queue: PR #3 v3 build, checks, A/B vs develop.
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
git -C openblas-fork fetch -q origin syrk-fewer-threads-v3
test -d openblas-sr-syrk3 || git -C openblas-fork worktree add -q --detach ../openblas-sr-syrk3 origin/syrk-fewer-threads-v3
lock "build syrk PR v3, checks" bash -c '
  set -e
  test -f builds/sr-syrk3/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sr-syrk3 builds/sr-syrk3
  git -C openblas-sr-syrk3 log --oneline -1
  scripts/check_patches.sh builds/sr-syrk3 > results/xeon-s1/check_syrk3.txt 2>&1'
lock "syrk PR v3 vs develop" scripts/run_syrk_v3.sh xeon-s1 4,8,16,22,43,86
