#!/bin/bash
# Experiment 1: MKL vs OpenBLAS vs BLIS map over sklearn-like shapes.
# usage: scripts/run_map.sh MACHINE THREADS   (THREADS comma-separated)
# Shapes are column-major MxNxK:TT. numpy equivalents (X row-major (n, d)):
#   X @ W, W (d, k)          -> k x n x d :NN
#   X.T @ Y, Y (n, k)        -> k x d x n :NT   (gram X.T @ X: d x d x n :NT)
#   X @ C.T, C (k, d)        -> k x n x d :TN   (KMeans / pairwise distances)
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
SHAPES=(
  16x16x16 32x32x32 64x64x64 128x128x128 256x256x256 512x512x512
  1024x1024x1024 2048x2048x2048 4096x4096x4096
  100x100x1000000:NT 1000x1000x100000:NT
  100x200000x100:NN 10x1000000x100:NN 50x100000x1000:NN 50x1000x100000:NT
  256x100000x50:TN 1000x20000x1000:TN
  2000x2000x200 200x2000x2000:NT 200x200x50000:TN
)
pixi run python scripts/sweep.py --out results/$MACHINE/map.csv \
  --threads "$THREADS" --variant openblas --variant openblas-omp --variant mkl \
  --variant blis --dtype d,s --rounds 2 --min-time 0.2 \
  --note "exp1 map: default library settings, no pinning" --shapes "${SHAPES[@]}"
