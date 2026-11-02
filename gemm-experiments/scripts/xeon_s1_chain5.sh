#!/bin/bash
# Fifth s1 queue: per-precision SWITCH_RATIO fix (exp 14).
cd "$(dirname "$0")/.."
lock() { benchlock --wait 14400 --tag "gemm: $1" s1 -- "${@:2}"; }
test -d openblas-fork || git clone -q https://github.com/cakedev0/OpenBLAS.git openblas-fork
git -C openblas-fork fetch -q origin develop syrk-fewer-threads switch-ratio-per-precision
mk() {  # name, then refs to merge on top of develop
  local d=openblas-$1; shift
  test -d $d && return 0
  git -C openblas-fork worktree add -q --detach ../$d origin/develop
  for r in "$@"; do git -C $d -c user.name=x -c user.email=x@x merge -q --no-edit origin/$r; done
}
mk sr-base
mk sr-fix switch-ratio-per-precision
mk sr-syrk syrk-fewer-threads
mk sr-syrk-fix syrk-fewer-threads switch-ratio-per-precision
lock "build OpenBLAS SWITCH_RATIO fix variants, checks" bash -c '
  set -e
  pixi run build
  for b in sr-base sr-fix sr-syrk sr-syrk-fix; do
    test -f builds/$b/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-$b builds/$b
  done
  scripts/check_patches.sh builds/sr-fix builds/sr-syrk-fix > results/xeon-s1/check_switch_ratio.txt 2>&1'
lock "SWITCH_RATIO fix A/B" scripts/run_switch_ratio.sh xeon-s1 8,22,43,64,86
