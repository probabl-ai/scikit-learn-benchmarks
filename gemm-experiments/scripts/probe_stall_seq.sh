#!/bin/bash
# Replay the exp22 shape list up to getrf 8000 in one gemmbench process at T
# threads on the first T cores, and sample thread placement during it.
set -uo pipefail
cd "$(dirname "$0")/.."
T=${T:-24}; B=${B:-tc2}
CPUS=$(pixi run python -c "
import sys; sys.path.insert(0, 'scripts'); from sweep import ${PIN:-compact_cpus}
print(','.join(map(str, ${PIN:-compact_cpus}(${CORES:-$T}))))")
S=()
for n in 10001 20000 50000 100000 200000 500000 1000000 2000000 5000000; do S+=(dot:$n axpy:$n); done
for n in 200 300 500 1000 2000 4000 8000; do S+=(symv:$n); done
for n in 300 500 1000 2000 4000 8000; do S+=(getrf:${n}x$n); done
OPENBLAS_NUM_THREADS=$T taskset -c $CPUS ./build/gemmbench --lib builds/$B/lib/libopenblas.so.0 \
  --dtype d --min-time 0.2 --threads $T --label $B "${S[@]}" > /tmp/stallseq.$$ 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  sleep 0.5
  line=$(tail -1 /tmp/stallseq.$$ | cut -d, -f3-7,11)
  snap=$(cat /proc/$pid/task/*/stat 2>/dev/null | awk '{sub(/.*\) /, ""); print $1, $37}')
  nr=$(echo "$snap" | awk '$1=="R"' | wc -l)
  shared=$(echo "$snap" | awk '{c[$2]++} END {for (k in c) if (c[k]>1) printf "%s:%d ", k, c[k]}')
  echo "last=[$line] running=$nr shared=[$shared]"
done
grep -v '^#' /tmp/stallseq.$$ | cut -d, -f3-7,11 | tail -8; rm -f /tmp/stallseq.$$
