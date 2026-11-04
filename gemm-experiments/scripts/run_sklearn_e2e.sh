#!/bin/bash
# Experiments 22b-23b: scikit-learn / scipy calls with OpenBLAS develop vs a
# patched build in a conda-forge numpy/scipy/sklearn env (pthreads OpenBLAS),
# interleaved, pinned to the first T cores of the slot.
# Each build gets a hard-linked copy of the env with the OpenBLAS file
# replaced: with LD_PRELOAD instead, numpy and scipy still load the env's own
# OpenBLAS through libblas.so.3 (RPATH wins), and its thread pool too.
# usage: [BASE=sr-base] [BUILD=tc] [ROUNDS=3] [OUT=sklearn_e2e]
#        scripts/run_sklearn_e2e.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/$1/${OUT:-sklearn_e2e}.csv
echo "round,label,threads,case,best_s,median_s,n_iter" > $OUT
CPUS=($(pixi run python -c "
import os
from pathlib import Path
seen, cpus = set(), []
for c in sorted(os.sched_getaffinity(0)):
    core = Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text()
    if core not in seen:
        seen.add(core); cpus.append(c)
print(' '.join(map(str, cpus)))"))
SRC=$(pixi exec -s python=3.12 -s numpy -s scipy -s scikit-learn -s "libblas=*=*openblas" \
  -s "libopenblas=*=*pthreads*" -- python -c "import sys; print(sys.prefix)")
LIBNAME=$(basename $(readlink -f $SRC/lib/libblas.so.3))
for b in ${BASE:-sr-base} ${BUILD:-tc}; do
  rm -rf envs/e2e-$b && mkdir -p envs && cp -al $SRC envs/e2e-$b
  rm envs/e2e-$b/lib/$LIBNAME && cp builds/$b/lib/libopenblas.so.0 envs/e2e-$b/lib/$LIBNAME
done
for r in $(seq ${ROUNDS:-3}); do
  for t in ${2//,/ }; do
    list=$(IFS=,; echo "${CPUS[*]:0:$t}")
    order="${BASE:-sr-base} ${BUILD:-tc}"; [ $((r % 2)) = 0 ] && order="${BUILD:-tc} ${BASE:-sr-base}"
    for b in $order; do
      OPENBLAS_NUM_THREADS=$t taskset -c $list \
        envs/e2e-$b/bin/python scripts/sklearn_e2e.py --label $b | sed "s/^/$r,/" >> $OUT
    done
  done
done
