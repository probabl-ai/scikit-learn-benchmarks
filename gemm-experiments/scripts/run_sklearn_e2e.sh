#!/bin/bash
# Experiment 22b: scikit-learn / scipy calls with OpenBLAS develop vs a patched
# build, swapped in with LD_PRELOAD into a conda-forge numpy/scipy/sklearn env
# (pthreads OpenBLAS), interleaved, pinned to the first T cores of the slot.
# usage: [BASE=sr-base] [BUILD=tc] [ROUNDS=3] scripts/run_sklearn_e2e.sh MACHINE THREADS
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
ENV=(pixi exec -s python=3.12 -s numpy -s scipy -s scikit-learn -s "libblas=*=*openblas" -s "libopenblas=*=*pthreads*" --)
for r in $(seq ${ROUNDS:-3}); do
  for t in ${2//,/ }; do
    list=$(IFS=,; echo "${CPUS[*]:0:$t}")
    order="${BASE:-sr-base} ${BUILD:-tc}"; [ $((r % 2)) = 0 ] && order="${BUILD:-tc} ${BASE:-sr-base}"
    for b in $order; do
      LD_PRELOAD=$PWD/builds/$b/lib/libopenblas.so.0 OPENBLAS_NUM_THREADS=$t taskset -c $list \
        "${ENV[@]}" python scripts/sklearn_e2e.py --label $b | sed "s/^/$r,/" >> $OUT
    done
  done
done
