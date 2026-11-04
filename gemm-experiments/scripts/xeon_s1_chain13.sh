#!/bin/bash
# Exp 27 on xeon-s1: split-K v3 (one chunk per thread when C is tiny) vs v2
# and develop, on the shapes where the chunk count changes.
set -uo pipefail
cd "$(dirname "$0")/.."
git -C openblas-fork fetch -q origin splitk-v3
test -d openblas-sk3 || git -C openblas-fork worktree add -q --detach ../openblas-sk3 origin/splitk-v3
test -f builds/sk3/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sk3 builds/sk3 >/dev/null
scripts/check_patches.sh builds/sk3 > results/xeon-s1/check_sk3.txt 2>&1
grep -c OK results/xeon-s1/check_sk3.txt; grep -v OK results/xeon-s1/check_sk3.txt | head
BUILD="sk2 sk3" OUT=sk3 ROUNDS=4 ONLY="^(syrk:)?(32|64|128|256|512)x([0-9]+x)?(5000|20000|50000):" scripts/run_splitk.sh xeon-s1 8,16,24,43,64,86
