#!/bin/bash
# usage: tsan_run.sh LABEL (venv-LABEL). Prints the number of TSan reports and their first frames.
W=$(cd "$(dirname "$0")" && pwd); cd "$W"; L=$1; shift
E=$W/.pixi/envs/default/lib
rm -f tsan-$L.log.*
env "$@" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_TOOL_LIBRARIES=$W/tool/libarcher.so ARCHER_OPTIONS="verbose=1" \
  TSAN_OPTIONS="ignore_noninstrumented_modules=1 exitcode=0 log_path=$W/tsan-$L.log history_size=4" \
  LD_LIBRARY_PATH=$E LD_PRELOAD=$E/libclang_rt.tsan-x86_64.so:$E/libomp.so \
  $W/venv-$L/bin/python $W/tsan_fit.py 2>&1 | tail -12 | cut -c1-120
n=$(cat tsan-$L.log.* 2>/dev/null | grep -c "WARNING: ThreadSanitizer")
echo "$L: $n TSan reports"
cat tsan-$L.log.* 2>/dev/null | grep -A3 -E "^  (Write|Read|Previous|Atomic)" | grep -E "#0" | sed 's/.*#0 //' | cut -c1-110 | sort | uniq -c | sort -rn | head -12
