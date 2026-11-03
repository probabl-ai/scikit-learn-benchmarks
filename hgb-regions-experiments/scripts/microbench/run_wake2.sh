#!/bin/bash
cd ~/work/gamer-run/wake
E=~/scikit-learn-benchmarks/.pixi/envs
$E/sklearn-dev/bin/gcc -O2 -fopenmp wake2.c -o wake2 || exit 1
GAPS="0 160 500 2000"
# ~per-thread working sets: 32 KB items -> 128 KB/thread (fits L2), 256 KB items -> 1 MB/thread
for cfg in "32 4" "256 1"; do
  set -- $cfg
  for mode in passive spin; do
    if [ $mode = spin ]; then X="GOMP_SPINCOUNT=300000"; else X=""; fi
    echo "== libgomp $mode"
    env $X OMP_NUM_THREADS=23 ./wake2 1000 $1 $2 $GAPS
  done
done
echo "== passive, 16 threads"
OMP_NUM_THREADS=16 ./wake2 1000 32 4 $GAPS
echo "== wake2 done $(date -Is)"
