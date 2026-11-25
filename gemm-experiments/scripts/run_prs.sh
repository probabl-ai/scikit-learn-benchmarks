#!/bin/bash
# Exp 31: each open fork PR against develop c2fc6cf5f, on the shapes of its
# description, one sweep per PR (develop and the PR interleaved).
# Builds: builds/pr-base (develop), builds/pr3, pr8, pr10, pr11.
# usage: [PRS="3 8 10 11"] [ROUNDS=3] [PIN=compact] scripts/run_prs.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
SO=libopenblas.so.0; PIN=${PIN:-compact}
[ "$(uname)" = Darwin ] && SO=libopenblas.0.dylib && PIN=none
MAXELEM=100000000  # max elements of one input (8e8 bytes in f64)

shapes_3() {  # syrk X.T @ X with few columns: develop falls back to one thread
  for n in 32 64 128 256 512 990; do
    for k in 5000 20000 100000; do
      [ $((n * k)) -le $MAXELEM ] && echo syrk:${n}x$k:N
    done
  done
}
shapes_10() {  # small C, long k
  for m in 32 64 128 256 512; do
    for k in 20000 200000 1000000; do
      [ $((m * k)) -le $MAXELEM ] && echo ${m}x${m}x$k:NT syrk:${m}x$k:N
    done
  done
  echo 10x200x1000000:NT 200x200x50000:TN 100x100x1000000:TN 1000x1000x100000:NT 2000x2000x2000
}
shapes_8() {
  echo dot:100000 dot:1000000 dot:10000000 axpy:100000 axpy:1000000 axpy:10000000
  echo getrf:1000x1000 getrf:2000x2000 getrf:4000x4000 getrf:20000x500
  echo gesv:2000x1 gesv:1000x1000 gesv:1000x10000 gesv:2000x20000
}
shapes_11() { echo symv:200 symv:500 symv:1000 symv:2000 symv:4000; }
# gemmbench times the routines other than gemm and syrk in double only.
declare -A DTYPES=([3]=d,s [10]=d,s [8]=d [11]=d)

for pr in ${PRS:-3 8 10 11}; do
  pixi run python scripts/sweep.py --out results/$1/prs/pr$pr.csv --threads "$2" \
    --variant "develop=builds/pr-base/lib/$SO,OPENBLAS_NUM_THREADS={T}" \
    --variant "pr$pr=builds/pr$pr/lib/$SO,OPENBLAS_NUM_THREADS={T}" \
    --dtype ${DTYPES[$pr]} --rounds ${ROUNDS:-3} --min-time 0.2 --pin $PIN \
    --note "exp31 cakedev0/OpenBLAS#$pr vs develop c2fc6cf5f" --shapes $(shapes_$pr)
done
