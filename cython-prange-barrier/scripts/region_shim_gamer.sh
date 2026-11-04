#!/bin/bash
# Run under benchlock on intel-gamer-gpu: region_shim_run.sh where Fix B loses (24T, P+E) and where it wins.
L=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
d=~/work/cython-prange/placement
bash "$d/region_shim_run.sh" mix 24 0-23 5 "$L"
bash "$d/region_shim_run.sh" eonly 16 8-23 5 "$L"
bash "$d/region_shim_run.sh" ponly 8 0-7 5 "$L"
