#!/bin/bash
# Experiment 4: A/B of the OpenBLAS patches against conda-forge OpenBLAS,
# the unpatched develop build and MKL.
# usage: [SWEEP_ARGS=...] scripts/run_patches.sh MACHINE THREADS [OUTNAME] [extra builds...]
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
OUT=${3:-patches}
shift 3 || shift $#
SHAPES=(
  100x100x1000000:NT 100x100x800000:NT 200x200x50000:TN 50x1000x100000:NT
  1000x1000x100000:NT
  100x200000x100:NN 10x1000000x100:NN 7x116203x100:NN 50x100000x1000:NN
  256x100000x50:TN
  256x256x256 1024x1024x1024 4096x4096x4096 2000x2000x200 200x2000x2000:NT
)
V=()
for b in base splitk l1 "$@"; do
  V+=(--variant "$b=builds/$b/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}")
done
# split-K forced to one chunk per thread, to see its potential at any T
V+=(--variant "splitk-force=builds/splitk/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T},OPENBLAS_GEMM_SPLITK={T}")
pixi run python scripts/sweep.py --out results/$MACHINE/$OUT.csv \
  --threads "$THREADS" --variant openblas --variant mkl "${V[@]}" \
  --dtype d,s --rounds 2 --min-time 0.3 \
  --note "exp4 OpenBLAS patches A/B (base = develop 800bb92a)" ${SWEEP_ARGS:---pin compact} --shapes "${SHAPES[@]}"
