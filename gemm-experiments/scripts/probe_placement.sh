#!/bin/bash
# Exp 11d: where do MKL's OpenMP threads run? Average number of running
# threads, distinct CPUs and distinct cores they occupy (20 ms samples while
# timing), plus GF/s, per runtime setting. Unpinned, like exp 11.
# usage: scripts/probe_placement.sh OUT.csv THREADS...
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; shift
L=.pixi/envs/default/lib/libmkl_rt.so
IOMP=$PWD/vendor/intel-openmp/libiomp5.so
declare -A V=(
  [mkl]=""
  [btinf]="KMP_BLOCKTIME=infinite"
  [close]="OMP_PROC_BIND=close OMP_PLACES=cores"
  [iomp]="LD_PRELOAD=$IOMP"
)
TMP=$(mktemp -p .)
echo "round,variant,threads,shape,gflops_median,samples,running,cpus,cores,involuntary_switches" > "$OUT"
for r in 0 1 2; do
for t in "$@"; do
  for shape in 4096x4096x4096 100x100x1000000:NT; do
    for v in mkl btinf close iomp; do
      res=$(env ${V[$v]} GEMMBENCH_SAMPLE_PLACEMENT=1 MKL_NUM_THREADS=$t build/gemmbench \
            --lib $L --threads $t --min-time 1.5 $shape 2>&1 >"$TMP" | grep -E "placement|affinity" || true)
      gf=$(cut -d, -f12 "$TMP")
      pl=$(sed -n 's/.*samples=\([0-9.]*\) running=\([0-9.]*\) cpus=\([0-9.]*\) cores=\([0-9.]*\).*/\1,\2,\3,\4/p' <<<"$res")
      isw=$(sed -n 's/.*involuntary_switches=\([0-9]*\).*/\1/p' <<<"$res")
      echo "$r,$v,$t,$shape,$gf,$pl,$isw" | tee -a "$OUT"
    done
  done
done
done
rm -f "$TMP"
