#!/bin/bash
# Experiment 22: thread-count patch (dot, axpy, symv, getrf, gesv) vs develop.
# usage: [BASE=sr-base] [BUILD=tc] [OUT=tc] [ROUNDS=3] [PIN=compact]
#        scripts/run_tc.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
SO=libopenblas.so.0; [ "$(uname)" = Darwin ] && SO=libopenblas.0.dylib
S=()
for n in 10001 20000 50000 100000 200000 500000 1000000 2000000 5000000; do S+=(dot:$n axpy:$n); done
for n in 200 300 500 1000 2000 4000 8000; do S+=(symv:$n); done
for n in 300 500 1000 2000 4000 8000; do S+=(getrf:${n}x$n); done
S+=(getrf:100000x20 getrf:1000000x20 getrf:20000x500 gesv:500x1 gesv:2000x1 gesv:4000x100
    syevd:200 syevd:1000 syevd:4000 syevr:1000 syevr:4000)
for b in ${BASE:-sr-base} ${BUILD:-tc}; do
  [ -n "${CHECK:-}" ] && for t in ${CHECK}; do
    OPENBLAS_NUM_THREADS=$t ./build/gemmbench --lib builds/$b/lib/$SO --check 1 "${S[@]}" | sed "s/^/$b,/"
  done
done
[ -n "${CHECK:-}" ] && exit 0
V=()
for b in ${BASE:-sr-base} ${BUILD:-tc}; do V+=(--variant "$b=builds/$b/lib/$SO,OPENBLAS_NUM_THREADS={T}"); done
pixi run python scripts/sweep.py --out results/$1/${OUT:-tc}.csv --threads "$2" "${V[@]}" \
  --dtype d --rounds ${ROUNDS:-3} --min-time 0.2 --pin ${PIN:-compact} \
  --note "exp22 thread-count patch vs develop" --shapes "${S[@]}"
