#!/bin/bash
# Exp 28 on intel-gamer-gpu: the effective long-term power limit is 65 W over
# ~32 s (intel-rapl-mmio), bursts up to 182-200 W. Does ~40 s of all-core
# work make 2000^3 slower on develop as well? Same shape repeated, in order.
set -euo pipefail
cd "$(dirname "$0")/.."
V=(--variant sr-base=builds/sr-base/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}
   --variant sk3=builds/sk3/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T})
heat=$(for i in $(seq 60); do printf '1000x1000x100000:NT '; done)
pixi run python scripts/sweep.py --out results/intel-gamer-gpu/power_budget.csv \
  --threads 24 "${V[@]}" --dtype d --rounds 2 --min-time 0.2 --pin compact \
  --note "exp28 power budget" --shapes 2000x2000x2000 $heat 2000x2000x2000
