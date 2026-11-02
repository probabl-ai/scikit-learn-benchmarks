#!/bin/bash
# Exp 19 on another machine: develop (fork) vs PR #3 v4, built here.
# usage: scripts/setup_syrk_v4_remote.sh MACHINE THREADS [extra env for run_syrk_v3.sh]
set -uo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1; THREADS=$2
SO=libopenblas.so.0; [ "$(uname)" = Darwin ] && SO=libopenblas.0.dylib
lock() { benchlock --wait 14400 --tag "gemm: $1" all -- "${@:2}"; }
test -d openblas-fork || git clone -q https://github.com/cakedev0/OpenBLAS.git openblas-fork
git -C openblas-fork fetch -q origin develop syrk-fewer-threads-v4
test -d openblas-sr-base || git -C openblas-fork worktree add -q --detach ../openblas-sr-base origin/develop
test -d openblas-sr-syrk4 || git -C openblas-fork worktree add -q --detach ../openblas-sr-syrk4 origin/syrk-fewer-threads-v4
lock "pixi install + build develop/syrk v4 OpenBLAS, checks" bash -c "
  set -e
  pixi install >/dev/null && pixi run build
  for b in sr-base sr-syrk4; do
    test -f builds/\$b/lib/$SO || JOBS=\${JOBS:-8} scripts/build_openblas.sh openblas-\$b builds/\$b
  done
  mkdir -p results/$MACHINE
  scripts/check_patches.sh builds/sr-syrk4 > results/$MACHINE/check_syrk4.txt 2>&1 || true"
lock "syrk PR v4 vs develop" env BUILD=sr-syrk4 OUT=syrk_v4 SMALL_MACHINE=1 "${@:3}" \
  scripts/run_syrk_v3.sh $MACHINE $THREADS
