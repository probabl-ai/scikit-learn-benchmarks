#!/bin/bash
# Exp 11e: syscalls made by MKL's OpenMP workers (strace -c over a fixed
# number of 4096^3 calls), per runtime setting. Timings under strace are not
# meaningful; only the syscall counts are.
# usage: scripts/probe_yield.sh OUTDIR THREADS...
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; shift
mkdir -p "$OUT"
L=.pixi/envs/default/lib/libmkl_rt.so
IOMP=$PWD/vendor/intel-openmp/libiomp5.so
declare -A V=(
  [mkl]=""
  [btinf]="KMP_BLOCKTIME=infinite"
  [close]="OMP_PROC_BIND=close OMP_PLACES=cores"
  [iomp]="LD_PRELOAD=$IOMP"
)
for t in "$@"; do
  for v in mkl btinf close iomp; do
    env ${V[$v]} MKL_NUM_THREADS=$t strace -f -c -o "$OUT/strace_${v}_T$t.txt" \
      build/gemmbench --lib $L --threads $t --min-time 100 --max-reps 20 4096x4096x4096 \
      > "$OUT/out_${v}_T$t.csv" 2> "$OUT/err_${v}_T$t.txt"
    echo "== $v T=$t"; head -12 "$OUT/strace_${v}_T$t.txt"
  done
done
