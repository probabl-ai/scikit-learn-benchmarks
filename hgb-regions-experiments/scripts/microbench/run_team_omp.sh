#!/bin/bash
cd ~/work/gamer-run/team; E=~/scikit-learn-benchmarks/.pixi/envs
for w in passive spin; do
  X=""; [ $w = spin ] && X="KMP_BLOCKTIME=200ms OMP_WAIT_POLICY=active"
  echo "== libomp $w"
  env $X LD_PRELOAD=$E/sklearn-dev-libomp/lib/libomp.so KMP_SETTINGS=1 ./team 23 4 2 5 2>&1 | grep -iE "BLOCKTIME|LIBRARY|WAIT" | head -3
  env $X LD_PRELOAD=$E/sklearn-dev-libomp/lib/libomp.so ./team 23 4 2000 5
  env $X LD_PRELOAD=$E/sklearn-dev-libomp/lib/libomp.so ./team 23 18 2000 5
done
