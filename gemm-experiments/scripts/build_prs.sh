#!/bin/bash
# Exp 31: build develop c2fc6cf5f and the heads of the four open fork PRs.
# usage: [JOBS=N] scripts/build_prs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
declare -A REF=([pr-base]=c2fc6cf5f784aab8eea95de3876d12f546cf787f
                [pr3]=89e62160fdc721430bb369e221c7b49e81ee7a82
                [pr8]=d851ca85f6fec94b3ec847e0e337e240de7d9109
                [pr10]=03536cdb11d1b435fbfac030976ce8d5747f1ab1
                [pr11]=21cbb8fc2e84e472c0f6f3c0adbff0c81a44785e)
test -d openblas-fork || git clone -q https://github.com/cakedev0/OpenBLAS.git openblas-fork
git -C openblas-fork fetch -q origin develop syrk-fewer-threads thread-counts splitk symv-fewer-threads
for b in pr-base pr3 pr8 pr10 pr11; do
  test -d openblas-$b || git -C openblas-fork worktree add -q --detach ../openblas-$b ${REF[$b]}
  test -f builds/$b/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-$b builds/$b
done
