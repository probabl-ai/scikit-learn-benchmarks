#!/bin/bash
# Experiment 8: KMeans-like usage, P application threads each calling a
# single-threaded sgemm/dgemm on its own 256-row chunk (gemmbench --callers).
# scikit-learn's Lloyd step calls X_chunk (256, d) @ centers.T (d, k), i.e.
# column-major k x 256 x d :TN. Library threads are set to 1.
# usage: scripts/run_callers.sh MACHINE CALLERS (comma-separated)
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
OUT=results/$MACHINE/callers.csv
mkdir -p results/$MACHINE
SHAPES=(1000x256x128:TN 100x256x784:TN 10x256x100:TN 256x256x256)
echo "label,dtype,m,n,k,trans,threads,reps,best_s,median_s,gflops_best,gflops_median,callers,round" > $OUT
E=.pixi/envs/default/lib
for round in 0 1; do
  for p in ${2//,/ }; do
    cpus=$(pixi run python -c "import sys; sys.path.insert(0, 'scripts'); import sweep; print(','.join(map(str, sweep.compact_cpus($p))))" | tail -1)
    for v in openblas mkl l1 base; do
      case $v in
        openblas) lib=$E/libopenblas.so.0 ;;
        mkl) lib=$E/libmkl_rt.so ;;
        *) lib=builds/$v/lib/libopenblas.so.0; [ -f $lib ] || continue ;;
      esac
      for dt in s d; do
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c $cpus build/gemmbench \
          --lib $lib --label $v --dtype $dt --threads 1 --callers $p --min-time 0.3 \
          "${SHAPES[@]}" 2>/dev/null | sed "s/$/,$p,$round/" | tee -a $OUT
      done
    done
  done
done
