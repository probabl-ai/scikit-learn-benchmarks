#!/bin/bash
# Concurrent callers: P application threads each call syrk/gemm with T library
# threads, all pinned to the first T cores of the allowed CPUs.
# usage: [ROUNDS=2] [DTYPES=d] [VARIANTS=..] [THREADS=..] [CALLERS=..] [PFX=cc-] [BENCH=build/gemmbench-cc]
#        run_callers_splitk.sh OUT.csv   (from the gemm-experiments directory)
set -uo pipefail
OUT=$1; PFX=${PFX-cc-}; BENCH=${BENCH:-build/gemmbench-cc}
SHAPES=(${SHAPES:-syrk:32x20000:N syrk:64x5000:N syrk:100x20000:N syrk:100x1000000:N syrk:200x100000:N syrk:512x20000:N
        64x64x20000:NT 128x128x5000:NT 200x200x50000:TN 1000x1000x2000:NN})
[ -f $OUT ] || echo "label,dtype,m,n,k,trans,threads,reps,best_s,median_s,gflops_best,gflops_median,callers,T,round" > $OUT
for round in $(seq 1 ${ROUNDS:-2}); do
  for T in ${THREADS:-16 43 86}; do
    cpus=$(pixi run python -c "import sys; sys.path.insert(0, 'scripts'); import sweep; print(','.join(map(str, sweep.compact_cpus($T))))" | tail -1)
    for P in ${CALLERS:-1 2 4 8}; do
      for dt in ${DTYPES:-d}; do
        for v in ${VARIANTS:-base sk v4a v4b}; do
          env OPENBLAS_NUM_THREADS=$T OMP_NUM_THREADS=$T ${ENVX:-} taskset -c $cpus $BENCH \
            --lib builds/$PFX$v/lib/libopenblas.so.0 --label $v${LSUF:-} --dtype $dt --threads $T --callers $P \
            --min-time ${MINTIME:-0.4} "${SHAPES[@]}" 2>/dev/null | sed "s/$/,$P,$T,$round/" >> $OUT
        done
      done
    done
  done
done
echo RUN_DONE
