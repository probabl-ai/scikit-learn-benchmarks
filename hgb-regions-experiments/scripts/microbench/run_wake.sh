#!/bin/bash
cd ~/work/gamer-run/wake
E=~/scikit-learn-benchmarks/.pixi/envs
$E/sklearn-dev/bin/gcc -O2 -fopenmp wake.c -o wake || exit 1
GAPS="0 20 50 100 160 300 500 1000 2000"
for rt in libgomp libomp; do
  if [ $rt = libomp ]; then PRE="$E/sklearn-dev-libomp/lib/libomp.so"; else PRE=""; fi
  for T in 23 16 8; do
    echo "== $rt passive (default)"
    LD_PRELOAD=$PRE OMP_NUM_THREADS=$T ./wake 2000 50 $GAPS
  done
  echo "== $rt spinning (GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms)"
  LD_PRELOAD=$PRE GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms OMP_NUM_THREADS=23 ./wake 2000 50 $GAPS
done
echo "== check runtime under LD_PRELOAD"
LD_PRELOAD=$E/sklearn-dev-libomp/lib/libomp.so OMP_DISPLAY_ENV=TRUE OMP_NUM_THREADS=2 ./wake 1 1 0 2>&1 | head -n 3
echo "== wake done $(date -Is)"
