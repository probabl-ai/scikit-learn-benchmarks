#!/bin/bash
# Experiment 14: DYNAMIC_ARCH builds use the single-precision SWITCH_RATIO /
# GEMM_PREFERRED_SIZE for double too (16 instead of 8 on SAPPHIRERAPIDS).
# A/B of the fix (fork branch switch-ratio-per-precision) alone and with the
# syrk thread-count fix (branch syrk-fewer-threads), all from the same base.
# usage: [SWEEP_ARGS=...] scripts/run_switch_ratio.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
SHAPES=(
  256x256x256 512x512x512 1024x1024x1024 4096x4096x4096 2000x2000x200
  100x200000x100:NN 7x116203x100:NN 10x1000000x100:NN 50x100000x1000:NN
  256x100000x50:TN 200x200x50000:TN 1000x1000x100000:NT
  syrk:990x463810:N syrk:1000x50000:N syrk:500x100000:N syrk:300x200000:N
  syrk:2000x100000:N
)
V=()
for b in sr-base sr-fix sr-syrk sr-syrk-fix; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
done
pixi run python scripts/sweep.py --out results/$MACHINE/switch_ratio.csv \
  --threads "$THREADS" "${V[@]}" --dtype d,s --rounds 3 --min-time 0.3 \
  --note "exp14 per-precision SWITCH_RATIO in DYNAMIC_ARCH" ${SWEEP_ARGS:---pin compact} \
  --shapes "${SHAPES[@]}"
