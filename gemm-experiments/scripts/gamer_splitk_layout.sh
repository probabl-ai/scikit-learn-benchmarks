#!/bin/bash
# Exp 28 on intel-gamer-gpu: is split-K's slowdown on develop's path (2000^3
# at 12-24T) from calling gemm_splitk, or from the code layout? sk3off = v3
# with the split-K calls unreachable (args.m < 0), same code otherwise.
set -uo pipefail
cd "$(dirname "$0")/.."
git -C openblas-fork fetch -q origin splitk-v3
test -d openblas-sk3 || git -C openblas-fork worktree add -q --detach ../openblas-sk3 origin/splitk-v3
if ! test -d openblas-sk3off; then
  git -C openblas-fork worktree add -q --detach ../openblas-sk3off origin/splitk-v3
  sed -i 's/if (args.nthreads > 1 \&\& gemm_splitk(/if (args.nthreads > 1 \&\& args.m < 0 \&\& gemm_splitk(/' openblas-sk3off/interface/gemm.c
  sed -i 's/if (args.nthreads > 1 \&\& syrk_splitk(/if (args.nthreads > 1 \&\& args.n < 0 \&\& syrk_splitk(/' openblas-sk3off/interface/syrk.c
  git -C openblas-sk3off diff --stat
fi
for b in sk3 sk3off; do
  test -f builds/$b/lib/libopenblas.so.0 || JOBS=16 scripts/build_openblas.sh openblas-$b builds/$b
done
env BUILD="sk2 sk3 sk3off" OUT=sk3_layout ROUNDS=6 \
  ONLY="^(2000x2000x2000|512x512x20000:NT|800x800x5000:NT|1000x1000x100000:NT)$" \
  scripts/run_splitk.sh intel-gamer-gpu 8,12,16,24
