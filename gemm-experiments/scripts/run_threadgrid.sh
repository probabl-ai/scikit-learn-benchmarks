#!/bin/bash
# Experiment 21: fine thread-count grid for the OpenBLAS routines that slow
# down with more threads in exp20, to find the best thread count per size.
# usage: [BUILD=sr-base] [OUT=threadgrid] [ROUNDS=2] scripts/run_threadgrid.sh MACHINE
set -euo pipefail
cd "$(dirname "$0")/.."
B=${BUILD:-sr-base}
OUT=${OUT:-threadgrid}
L12=()
for n in 10001 20000 50000 100000 200000 500000 1000000 2000000 5000000; do
  L12+=(dot:$n axpy:$n)
done
for n in 300 500 1000 2000 4000; do L12+=(symv:$n); done
LAPACK=()
for n in 500 1000 2000 4000 8000; do
  LAPACK+=(getrf:${n}x$n potrf:$n trsm:${n}x100)
done
LAPACK+=(trsm:2000x10 trsm:2000x1000 trsm:8000x10 trsm:8000x1000)
run() {
  pixi run python scripts/sweep.py --out results/$1/$OUT-$2.csv --threads "$3" \
    --variant "$B=builds/$B/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}" \
    --dtype d --rounds ${ROUNDS:-2} --min-time 0.2 --pin compact \
    --note "exp21 thread grid $2" --shapes "${@:4}"
}
run "$1" l12 1,2,3,4,6,8,12,16,24,32,48,64,86 "${L12[@]}"
run "$1" lapack 4,8,12,16,24,32,48,64,86 "${LAPACK[@]}"
