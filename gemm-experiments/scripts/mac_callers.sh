#!/bin/bash
# Concurrent single-threaded callers: P threads each run their own GEMM for
# the same window (gemmbench --callers P). On macOS this measures how many
# AMX units Accelerate can feed and what the E cores add to NEON throughput.
#   scripts/mac_callers.sh --out FILE.csv ROUNDS CALLERS DTYPES MIN_TIME SHAPE...
# e.g. scripts/mac_callers.sh --out x.csv 2 1,2,4,5,8 d,s 2 512x512x512
# reps/min_time*flops is the aggregate rate; gflops_* from per-call times
# assume every caller is equally fast, which is false once E cores join.
set -eu
out=$2; rounds=$3; callers=$4; dtypes=$5; mt=$6; shift 6
OB=.pixi/envs/default/lib/libopenblas.0.dylib
AC=/System/Library/Frameworks/Accelerate.framework/Accelerate
echo "round,label,dtype,m,n,k,trans,threads,reps,best_s,median_s,gflops_best,gflops_median" > "$out"
for r in $(seq "$rounds"); do
  for P in ${callers//,/ }; do
    for dt in ${dtypes//,/ }; do
      OPENBLAS_NUM_THREADS=1 ./build/gemmbench --lib $OB --threads 1 --label openblas \
        --dtype "$dt" --callers "$P" --min-time "$mt" "$@" 2>/dev/null | sed "s/^/$r,/" >> "$out"
      VECLIB_MAXIMUM_THREADS=1 ./build/gemmbench --lib $AC --label accelerate \
        --dtype "$dt" --callers "$P" --min-time "$mt" "$@" 2>/dev/null | sed "s/^/$r,/" >> "$out"
      ./build/gemmbench --lib $AC --label accelerate-mt \
        --dtype "$dt" --callers "$P" --min-time "$mt" "$@" 2>/dev/null | sed "s/^/$r,/" >> "$out"
      tail -n 3 "$out"
    done
  done
done
