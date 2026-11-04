#!/bin/bash
# Exp 24 on another machine: develop (fork) vs the split-K branch, built here.
# usage: scripts/setup_splitk_remote.sh MACHINE THREADS [extra env for run_splitk.sh]
set -uo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1; THREADS=$2; SK=${SK:-sk2}
SO=libopenblas.so.0; [ "$(uname)" = Darwin ] && SO=libopenblas.0.dylib
lock() { benchlock --wait 14400 --tag "gemm: $1" all -- "${@:2}"; }
test -d openblas-fork || git clone -q https://github.com/cakedev0/OpenBLAS.git openblas-fork
git -C openblas-fork fetch -q origin develop splitk
test -d openblas-sr-base || git -C openblas-fork worktree add -q --detach ../openblas-sr-base origin/develop
test -d openblas-$SK || git -C openblas-fork worktree add -q --detach ../openblas-$SK origin/splitk
lock "build split-K OpenBLAS, checks" bash -c "
  set -e
  pixi install >/dev/null && pixi run build
  for b in sr-base $SK; do
    test -f builds/\$b/lib/$SO || JOBS=\${JOBS:-8} scripts/build_openblas.sh openblas-\$b builds/\$b
  done
  mkdir -p results/$MACHINE
  scripts/check_patches.sh builds/$SK > results/$MACHINE/check_$SK.txt 2>&1 || true"
lock "split-K vs develop" env BUILD=$SK OUT=$SK "${@:3}" scripts/run_splitk.sh $MACHINE $THREADS
