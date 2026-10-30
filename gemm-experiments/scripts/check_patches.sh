#!/bin/bash
# Correctness of the patched OpenBLAS builds against a naive reference.
# usage: scripts/check_patches.sh BUILD_DIR...   (e.g. builds/splitk builds/l1)
set -euo pipefail
cd "$(dirname "$0")/.."
SHAPES=(37x45x2000 37x45x2000:NT 37x45x2000:TN 37x45x2000:TT 100x100x200000:NT
        10x100000x100 100x20000x100 7x50000x100:TN 50x30000x300:NT 513x257x129)
for b in "$@"; do
  for force in 0 7 ""; do
    for beta in 0 0.5; do
      for dt in d s; do
        OPENBLAS_NUM_THREADS=8 OPENBLAS_GEMM_SPLITK=$force build/gemmbench \
          --lib $b/lib/libopenblas.so.0 --label $b --dtype $dt --beta $beta \
          --check 1 "${SHAPES[@]}" 2>/dev/null | sed "s|^|$b splitk=${force:-auto} |"
      done
    done
  done
done
