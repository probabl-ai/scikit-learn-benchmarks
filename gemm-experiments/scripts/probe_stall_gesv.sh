#!/bin/bash
# gesv 2000 then gesv 4000x100 at T threads in one gemmbench process, which
# stalls (~2 s instead of 62 ms) in some runs on xeon-s1 at T=24. Variants:
# compact mask, compact mask with workers sleeping later, mask with siblings.
set -uo pipefail
cd "$(dirname "$0")/.."
T=${T:-24}
mask() { pixi run python -c "
import sys; sys.path.insert(0, 'scripts'); from sweep import $1
print(','.join(map(str, $1($T))))"; }
C=$(mask compact_cpus); S=$(mask compact_cpus_smt)
for rep in $(seq ${REPS:-4}); do
  for b in sr-base ${BUILD:-tc4}; do
    for v in compact timeout30 smt; do
      cpus=$C; extra=""
      [ $v = timeout30 ] && extra="OPENBLAS_THREAD_TIMEOUT=30"
      [ $v = smt ] && cpus=$S
      printf "%-8s %-9s " $b $v
      env $extra OPENBLAS_NUM_THREADS=$T taskset -c $cpus ./build/gemmbench --lib builds/$b/lib/libopenblas.so.0 \
        --dtype d --min-time 0.2 --threads $T --label $b gesv:2000x1 gesv:4000x100 2>/dev/null \
        | grep -v '^#' | awk -F, '{printf " %s=%.1fms", $7":"$4"x"$6, $11*1e3} END {print ""}'
    done
  done
done
