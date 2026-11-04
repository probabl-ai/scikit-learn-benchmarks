#!/bin/bash
# probe_colocation.py with T OpenBLAS threads pinned to the first T cores
# (the sampler thread makes T+1 threads in the process, all sleeping but one).
set -uo pipefail
cd "$(dirname "$0")/.."
T=${T:-24}
CPUS=$(pixi run python -c "
import sys; sys.path.insert(0, 'scripts'); from sweep import compact_cpus
print(','.join(map(str, compact_cpus(${CORES:-$T}))))")
ENV=(pixi exec -s python=3.12 -s numpy -s scipy -s scikit-learn -s "libblas=*=*openblas" -s "libopenblas=*=*pthreads*" --)
for rep in $(seq ${REPS:-4}); do
  for b in ${BUILDS:-sr-base}; do
    echo "== $b T=$T cores=${CORES:-$T} rep=$rep"
    LD_PRELOAD=$PWD/builds/$b/lib/libopenblas.so.0 OPENBLAS_NUM_THREADS=$T taskset -c $CPUS \
      "${ENV[@]}" python scripts/probe_colocation.py ${N:-1000}
  done
done
