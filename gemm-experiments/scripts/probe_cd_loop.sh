#!/bin/bash
# probe_cd_loop.py over thread counts, pinned to the first CORES cores.
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
B=${B:-sr-base}
for rep in 1 2; do
for t in ${THREADS:-38 39 40 41 42 43}; do
  list=$(IFS=,; echo "${CPUS[*]:0:${CORES:-43}}")
  printf "%s T=%s cores=%s args=%s " $B $t ${CORES:-43} "${ARGS:-1000000 20}"
  LIB=$PWD/builds/$B/lib/libopenblas.so.0 OPENBLAS_NUM_THREADS=$t taskset -c $list \
    ${CMD:-pixi run python scripts/probe_cd_loop.py ${ARGS:-1000000 20}}
done
done
