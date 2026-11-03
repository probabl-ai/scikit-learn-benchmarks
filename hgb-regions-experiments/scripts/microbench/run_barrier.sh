#!/bin/bash
cd ~/work/gamer-run/barrier; E=~/scikit-learn-benchmarks/.pixi/envs
T=${T:-23}
for rt in gomp omp; do
  P=""; [ $rt = omp ] && P="LD_PRELOAD=$E/sklearn-dev-libomp/lib/libomp.so"
  for w in passive spin; do
    X=""; [ $w = spin ] && X="GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms OMP_WAIT_POLICY=active"
    echo "== $rt $w"
    for item in 1 10; do for outer in 30 300; do
      env $P $X OMP_NUM_THREADS=$T ./barrier 4 $item $outer 2000 0 30 300
    done; done
  done
done
