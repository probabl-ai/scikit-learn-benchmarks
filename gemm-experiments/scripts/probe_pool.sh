#!/bin/bash
# Experiment 22c: is a level-1 call slower when it uses fewer threads than
# OpenBLAS's pool (OPENBLAS_NUM_THREADS) or than the CPUs it may run on?
# Develop build only; MASK cores (compact, physical) x POOL threads.
# usage: [BUILD=sr-base] scripts/probe_pool.sh
set -euo pipefail
cd "$(dirname "$0")/.."
B=${BUILD:-sr-base}
CPUS=($(pixi run python -c "
import os
from pathlib import Path
seen, cpus = set(), []
for c in sorted(os.sched_getaffinity(0)):
    core = Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text()
    if core not in seen:
        seen.add(core); cpus.append(c)
print(' '.join(map(str, cpus)))"))
for r in 1 2 3; do
  for mask in 64 80 86; do
    for pool in 40 64 80 86; do
      [ $pool -gt $mask ] && continue
      list=$(IFS=,; echo "${CPUS[*]:0:$mask}")
      OPENBLAS_NUM_THREADS=$pool taskset -c $list ./build/gemmbench --lib builds/$B/lib/libopenblas.so.0 \
        --min-time 0.2 --label "r$r-mask$mask" dot:1000000 dot:2000000 axpy:2000000 syevd:200 2>/dev/null
    done
  done
done
