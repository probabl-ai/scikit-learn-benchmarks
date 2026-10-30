#!/bin/bash
# Exp 11b: do MKL's OpenMP workers sleep inside/between calls? Voluntary
# context switches per call for each runtime setting (one shape per process,
# fixed number of calls, unpinned).
# usage: scripts/probe_sleep.sh OUT.csv THREADS...
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; shift
L=.pixi/envs/default/lib/libmkl_rt.so
IOMP=$PWD/vendor/intel-openmp/libiomp5.so
declare -A V=(
  [mkl]=""
  [bt200ms]="KMP_BLOCKTIME=200ms"
  [bt1s]="KMP_BLOCKTIME=1000"
  [btinf]="KMP_BLOCKTIME=infinite"
  [active]="OMP_WAIT_POLICY=active"
  [noyield]="KMP_USE_YIELD=0"
  [close]="OMP_PROC_BIND=close OMP_PLACES=cores"
  [iomp]="LD_PRELOAD=$IOMP"
)
TMP=$(mktemp -p .)
echo "variant,threads,shape,calls,gflops_median,threads_seen,voluntary_switches,switches_per_call_per_thread" > "$OUT"
for r in 0 1; do
for t in "$@"; do
  for shape in 256x256x256 4096x4096x4096 100x100x1000000:NT; do
    calls=2000; [ $shape = 4096x4096x4096 ] && calls=40; [ $shape = 100x100x1000000:NT ] && calls=200
    for v in mkl bt200ms bt1s btinf active noyield close iomp; do
      res=$(env ${V[$v]} MKL_NUM_THREADS=$t build/gemmbench --lib $L --threads $t \
            --min-time 1000 --max-reps $calls $shape 2>&1 >"$TMP" | grep affinity || true)
      gf=$(cut -d, -f12 "$TMP")
      th=$(sed -n 's/.*threads=\([0-9]*\).*/\1/p' <<<"$res")
      sw=$(sed -n 's/.*voluntary_switches=\([0-9]*\).*/\1/p' <<<"$res")
      per=$(awk -v s=$sw -v c=$calls -v t=$th 'BEGIN{printf "%.2f", s/c/t}')
      echo "$v,$t,$shape,$calls,$gf,$th,$sw,$per" | tee -a "$OUT"
    done
  done
done
done
rm -f "$TMP"
