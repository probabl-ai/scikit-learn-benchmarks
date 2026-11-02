#!/bin/bash
# Ninth s1 queue: PR #3 v4 (driver-only, work and minimum-thread rules).
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
git -C openblas-fork fetch -q origin syrk-fewer-threads-v4
test -d openblas-sr-syrk4 || git -C openblas-fork worktree add -q --detach ../openblas-sr-syrk4 origin/syrk-fewer-threads-v4
lock "build syrk PR v4, checks" bash -c '
  set -e
  test -f builds/sr-syrk4/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sr-syrk4 builds/sr-syrk4
  git -C openblas-sr-syrk4 log --oneline -1
  scripts/check_patches.sh builds/sr-syrk4 > results/xeon-s1/check_syrk4.txt 2>&1'
BUILD=sr-syrk4 OUT=syrk_v4 lock "syrk PR v4 vs develop" scripts/run_syrk_v3.sh xeon-s1 4,8,16,22,43,86
