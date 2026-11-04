#!/bin/bash
# Run under benchlock on intel-gamer-gpu: main_fit_ab.sh with libgomp, then with libomp.
cd ~/work/cython-prange
libomp=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
test -f "$libomp" || { echo "missing $libomp"; exit 1; }
bash main_fit_ab.sh gamer-gomp 24 5
bash main_fit_ab.sh gamer-omp 24 5 "LD_PRELOAD=$libomp"
