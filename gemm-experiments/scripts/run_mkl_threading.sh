#!/bin/bash
# Experiment 3: why MKL stops scaling on a Xeon socket (s0/s1 slot).
# OpenMP placement and MKL_DYNAMIC variants, with OpenBLAS as reference.
# usage: scripts/run_mkl_threading.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
SHAPES=(
  64x64x64 128x128x128 256x256x256 512x512x512 1024x1024x1024 4096x4096x4096
  100x100x1000000:NT 7x116203x100:NN 100x200000x100:NN
)
pixi run python scripts/sweep.py --out results/$MACHINE/mkl_threading.csv \
  --threads "$THREADS" --variant openblas --variant mkl \
  --variant "mkl-close:mkl,OMP_PROC_BIND=close,OMP_PLACES=cores" \
  --variant "mkl-spread:mkl,OMP_PROC_BIND=spread,OMP_PLACES=cores" \
  --variant "mkl-nodyn:mkl,MKL_DYNAMIC=FALSE" \
  --variant "mkl-passive:mkl,OMP_WAIT_POLICY=passive" \
  --dtype d --rounds 2 --min-time 0.3 \
  --note "exp3 MKL threading/placement variants" ${SWEEP_ARGS:-} --shapes "${SHAPES[@]}"
