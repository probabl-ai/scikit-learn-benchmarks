#!/bin/bash
# Experiment 20: thread scaling of the other float64 BLAS/LAPACK routines
# scikit-learn calls (level 1 in coordinate descent, LAPACK in Ridge, PCA,
# LinearRegression, GaussianProcess, randomized_svd), OpenBLAS develop vs MKL
# on Intel's OpenMP runtime.
# usage: [BUILD=sr-base] [OUT=routines] [ROUNDS=2] [GROUP=l12|lapack|all]
#        scripts/run_routines.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
L12=()
for n in 20000 100000 1000000 10000000; do
  L12+=(dot:$n axpy:$n nrm2:$n asum:$n scal:$n)
done
L12+=(ger:100000x20 ger:1000000x10 symv:500 symv:2000 symv:8000 trsv:2000 trsv:8000
      trsm:2000x10 trsm:2000x2000 trsm:8000x100)
LAPACK=(potrf:500 potrf:2000 potrf:8000 posv:2000x1 posv:8000x1 gesv:2000x1
        getrf:100000x20 getrf:1000000x20 getrf:2000x2000 getrf:8000x8000
        geqrf:100000x20 geqrf:1000000x20 geqrf:100000x200 geqrf:2000x2000
        syevd:200 syevd:1000 syevd:4000 syevr:1000 syevr:4000
        gesdd:100000x100 gesdd:100000x500 gesdd:2000x2000
        gelsd:100000x100x1 gelsd:100000x500x1 gelsd:1000000x50x1)
case ${GROUP:-all} in
  l12) SHAPES=("${L12[@]}") ;;
  lapack) SHAPES=("${LAPACK[@]}") ;;
  *) SHAPES=("${L12[@]}" "${LAPACK[@]}") ;;
esac
IOMP=$PWD/vendor/intel-openmp/libiomp5.so
B=${BUILD:-sr-base}
pixi run python scripts/sweep.py --out results/$1/${OUT:-routines}.csv --threads "$2" \
  --variant "$B=builds/$B/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}" \
  --variant "mkl-iomp:mkl,LD_PRELOAD=$IOMP" \
  --dtype d --rounds ${ROUNDS:-2} --min-time 0.2 --pin compact \
  --note "exp20 routine scaling ${GROUP:-all}" --shapes "${SHAPES[@]}"
