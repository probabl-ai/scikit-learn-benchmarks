#!/bin/bash
# Exp 24: split-K for gemm and syrk (fork branch splitk) vs develop, on a grid
# around the trigger (m * n < T * 128^2, k / T >= 256), f32 and f64.
# usage: [BASE=sr-base] [BUILD=sk] [OUT=splitk] [ROUNDS=3] [PIN=compact] [MAXELEM=1e8]
#        scripts/run_splitk.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
SO=libopenblas.so.0; [ "$(uname)" = Darwin ] && SO=libopenblas.0.dylib
MAXELEM=${MAXELEM:-100000000}  # max elements of one input (8e8 bytes in f64)
S=()
for m in 32 64 128 256 512 800 1100; do
  for k in 5000 20000 50000 200000 1000000; do
    [ $((m * k)) -le ${MAXELEM%.*} ] || continue
    S+=(${m}x${m}x$k:NT syrk:${m}x$k:N)
  done
done
S+=(10x200x1000000:NT 50x1000x100000:NT 200x200x50000:TN 100x100x1000000:TN
    syrk:300x200000:T syrk:100x1000000:T 2000x2000x2000 1000x1000x100000:NT)
[ -n "${ONLY:-}" ] && S=($(printf '%s\n' "${S[@]}" | grep -E "$ONLY"))
V=()
for b in ${BASE:-sr-base} ${BUILD:-sk}; do V+=(--variant "$b=builds/$b/lib/$SO,OPENBLAS_NUM_THREADS={T}"); done
pixi run python scripts/sweep.py --out results/$1/${OUT:-splitk}.csv --threads "$2" "${V[@]}" \
  --dtype d,s --rounds ${ROUNDS:-3} --min-time 0.2 --pin ${PIN:-compact} \
  --note "exp24 split-K vs develop" --shapes "${S[@]}"
