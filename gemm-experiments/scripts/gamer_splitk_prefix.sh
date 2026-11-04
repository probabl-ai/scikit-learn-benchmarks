#!/bin/bash
# Exp 28 on intel-gamer-gpu: which earlier call in the same process makes
# 2000^3 slower with v3? Each sweep times one prefix shape, then 2000^3.
set -euo pipefail
cd "$(dirname "$0")/.."
V=(--variant sr-base=builds/sr-base/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}
   --variant sk3=builds/sk3/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T})
for pre in 32x32x200000:NT syrk:32x200000:N 1100x1100x5000:NT syrk:100x1000000:T; do
  pixi run python scripts/sweep.py --out results/intel-gamer-gpu/sk3_prefix_${pre//:/_}.csv \
    --threads 24 "${V[@]}" --dtype d,s --rounds 3 --min-time 0.2 --pin compact \
    --note "exp28 prefix $pre" --shapes $pre 2000x2000x2000
done
