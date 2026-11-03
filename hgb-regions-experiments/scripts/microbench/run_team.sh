#!/bin/bash
cd ~/work/gamer-run/team; E=~/scikit-learn-benchmarks/.pixi/envs
for rt in gomp omp; do
  if [ $rt = omp ]; then L=$E/sklearn-dev-libomp/lib; else L=$E/sklearn-dev/lib; fi
  for w in passive spin; do
    X=""; [ $w = spin ] && X="GOMP_SPINCOUNT=300000 KMP_BLOCKTIME=200ms OMP_WAIT_POLICY=active"
    echo "== $rt $w ($(LD_LIBRARY_PATH=$L ldd ./team | grep -o "lib[g]*omp[^ ]* => [^ ]*"))"
    env $X LD_LIBRARY_PATH=$L ./team 23 4 2000 5; env $X LD_LIBRARY_PATH=$L ./team 23 18 2000 5
  done
done
