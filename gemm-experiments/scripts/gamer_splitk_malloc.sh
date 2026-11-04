#!/bin/bash
# Exp 28 on intel-gamer-gpu: develop's path is slower with split-K only when
# split shapes ran earlier in the same process. Suspect: split-K's freed
# buffers raise glibc's dynamic mmap threshold, so later matrices come from
# the heap instead of fresh mmaps. Two checks, develop vs v3:
# - the slow shapes alone in their process (sk3_alone.csv),
# - the full grid with the threshold fixed at its default (sk3_mmapfix.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
env BUILD=sk3 OUT=sk3_alone ROUNDS=4 ONLY="^(2000x2000x2000|1000x1000x100000:NT)$" \
  scripts/run_splitk.sh intel-gamer-gpu 16,24
env GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072 BUILD=sk3 OUT=sk3_mmapfix ROUNDS=3 \
  scripts/run_splitk.sh intel-gamer-gpu 16,24
