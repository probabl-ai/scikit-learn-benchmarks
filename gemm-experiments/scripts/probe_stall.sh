#!/bin/bash
# Look for scheduler stalls in threaded calls: time, context switches and
# migrations of one gemmbench process per build, pinned to the first T cores.
# usage: [T=24] [BUILDS="sr-base tc2"] [REPS=3] scripts/probe_stall.sh SHAPE...
set -uo pipefail
cd "$(dirname "$0")/.."
T=${T:-24}
CPUS=$(pixi run python -c "
import sys; sys.path.insert(0, 'scripts'); from sweep import compact_cpus
print(','.join(map(str, compact_cpus($T))))")
for rep in $(seq ${REPS:-3}); do
  for b in ${BUILDS:-sr-base tc2}; do
    OPENBLAS_NUM_THREADS=$T taskset -c $CPUS ./build/gemmbench --lib builds/$b/lib/libopenblas.so.0 \
      --dtype d --min-time ${MIN_TIME:-0.5} --threads $T --label $b "$@" 2>&1 \
      | grep -E "^$b,|affinity" | sed 's/^# affinity: //' | cut -d, -f1,3-6,10,11 | paste -sd' '
  done
done
