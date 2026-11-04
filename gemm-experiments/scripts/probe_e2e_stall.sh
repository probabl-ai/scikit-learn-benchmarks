#!/bin/bash
# sklearn_e2e.py cases with one build at T threads, first T cores, to find
# which preceding calls make a later one stall.
# usage: [T=24] [BUILDS="sr-base tc2"] scripts/probe_e2e_stall.sh ONLY...
set -uo pipefail
cd "$(dirname "$0")/.."
T=${T:-24}
CPUS=$(pixi run python -c "
import sys; sys.path.insert(0, 'scripts'); from sweep import compact_cpus
print(','.join(map(str, compact_cpus($T))))")
ENV=(pixi exec -s python=3.12 -s numpy -s scipy -s scikit-learn -s "libblas=*=*openblas" -s "libopenblas=*=*pthreads*" --)
for only in "$@"; do
  for b in ${BUILDS:-sr-base tc2}; do
    echo "== $b only='$only'"
    LD_PRELOAD=$PWD/builds/$b/lib/libopenblas.so.0 OPENBLAS_NUM_THREADS=$T taskset -c $CPUS \
      "${ENV[@]}" python scripts/sklearn_e2e.py --label $b --only "$only" | cut -d, -f3-5
  done
done
