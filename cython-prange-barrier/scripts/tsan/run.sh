#!/bin/bash
# In the pixi env, under benchlock. Run cases.py under TSan and Archer, summarize the reports. usage: run.sh LABEL [KEY=VAL ...]
W=$(cd "$(dirname "$0")" && pwd); cd "$W"; L=$1; shift
E=$CONDA_PREFIX/lib
mkdir -p tool logs
[ -f tool/libarcher.so ] || cp "$(ls "$E"/libarcher.so* | head -1)" tool/libarcher.so
rm -f logs/tsan-$L.log.*
(cd /tmp && env "$@" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_TOOL_LIBRARIES=$W/tool/libarcher.so ARCHER_OPTIONS="verbose=1" \
  TSAN_OPTIONS="ignore_noninstrumented_modules=1 exitcode=0 log_path=$W/logs/tsan-$L.log history_size=4" \
  LD_LIBRARY_PATH=$E LD_PRELOAD=$E/libclang_rt.tsan-x86_64.so:$E/libomp.so \
  timeout 1500 "$W/venv-$L/bin/python" "$W/cases.py" > "$W/logs/out-$L.txt" 2>&1; echo "$L exit=$?")
cut -c1-100 "logs/out-$L.txt" | tail -22
echo "$L: $(cat logs/tsan-$L.log.* 2>/dev/null | grep -c 'WARNING: ThreadSanitizer') TSan reports"
cat logs/tsan-$L.log.* 2>/dev/null | grep -E '^SUMMARY' | sed 's/^SUMMARY: ThreadSanitizer: //' | sed -E 's# /[^ ]*/# #' | cut -c1-120 | sort | uniq -c | sort -rn | head -25
