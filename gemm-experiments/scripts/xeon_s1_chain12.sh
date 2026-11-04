#!/bin/bash
# Exp 26 on xeon-s1: split-K v2 (24 MiB buffers, fewer chunks only for tiny
# C) vs develop.
set -uo pipefail
cd "$(dirname "$0")/.."
git -C openblas-fork fetch -q origin splitk
test -d openblas-sk2 || git -C openblas-fork worktree add -q --detach ../openblas-sk2 origin/splitk
test -f builds/sk2/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sk2 builds/sk2 >/dev/null
scripts/check_patches.sh builds/sk2 > results/xeon-s1/check_sk2.txt 2>&1
grep -c OK results/xeon-s1/check_sk2.txt; grep -v OK results/xeon-s1/check_sk2.txt | head
BUILD=sk2 OUT=sk2 scripts/run_splitk.sh xeon-s1 8,16,24,43,64,86
