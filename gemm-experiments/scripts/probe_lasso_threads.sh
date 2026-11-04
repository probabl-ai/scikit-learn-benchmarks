#!/bin/bash
# Lasso 1000000x20 time vs OpenBLAS thread count, develop only: once with the
# process pinned to the first T cores, once pinned to the first 43 cores.
set -euo pipefail
cd "$(dirname "$0")/.."
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
B=${B:-sr-base}
for t in ${THREADS:-24 32 36 38 39 40 41 42 43 44 46 48}; do
  for mask in $t 43; do
    list=$(IFS=,; echo "${CPUS[*]:0:$mask}")
    printf "%s T=%s cores=%s " $B $t $mask
    LD_PRELOAD=$PWD/builds/$B/lib/libopenblas.so.0 OPENBLAS_NUM_THREADS=$t taskset -c $list \
      "${ENV[@]}" python scripts/sklearn_e2e.py --label x --only "Lasso 1000000" | cut -d, -f4-
  done
done
