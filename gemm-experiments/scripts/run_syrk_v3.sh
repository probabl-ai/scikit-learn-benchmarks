#!/bin/bash
# Experiment 19: PR #3 v3 (single precision stays on one thread below 8
# threads) vs develop, f32 and f64, on a grid of small and large syrk.
# usage: [BUILD=sr-syrk3] [OUT=syrk_v3] [PIN=compact|none] [ROUNDS=2] [SMALL_MACHINE=1]
#        scripts/run_syrk_v3.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
SO=libopenblas.so.0; [ "$(uname)" = Darwin ] && SO=libopenblas.0.dylib
SHAPES=()
for n in 32 64 96 128 160 192 256 384 512; do
  for k in 1000 5000 20000 100000; do SHAPES+=(syrk:${n}x$k:N); done
done
SHAPES+=(syrk:64x2000:N syrk:990x2000:N syrk:1000x50000:N syrk:300x200000:N)
if [ -z "${SMALL_MACHINE:-}" ]; then
  SHAPES+=(syrk:990x463810:N syrk:2000x100000:N)
else
  # few features, many samples: below 2 x threads on VORTEX (switch_ratio 2)
  SHAPES+=(syrk:8x1000000:N syrk:10x1000000:N syrk:12x100000:N syrk:15x1000000:N
           syrk:20x100000:N)
fi
V=()
for b in sr-base ${BUILD:-sr-syrk3}; do
  V+=(--variant "$b=builds/$b/lib/$SO,OPENBLAS_NUM_THREADS={T}")
done
pixi run python scripts/sweep.py --out results/$1/${OUT:-syrk_v3}.csv --threads "$2" "${V[@]}" \
  --dtype d,s --rounds ${ROUNDS:-2} --min-time 0.2 --pin ${PIN:-compact} \
  --note "exp19 syrk PR v3 vs develop" --shapes "${SHAPES[@]}"
