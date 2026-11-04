#!/bin/bash
# Exp 18, libomp on gamer (the first attempt silently used libgomp). Under benchlock.
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
OMP=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
LD_PRELOAD=$OMP venv-clean/bin/python -c "
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as R
from threadpoolctl import threadpool_info
R(max_iter=1).fit(np.random.rand(100, 2), np.random.rand(100))
print('runtimes', [p['prefix'] for p in threadpool_info() if p['user_api'] == 'openmp'])"
venv-main/bin/python bench.py --profile exp18-gamer-omp --threads 24 --variants pr,pr+fuse3,abs+fuse3,clean:pr --n-runs 1 --reps 5 --phases --env LD_PRELOAD=$OMP --out exp18-gamer-omp2.jsonl
echo "### all done"
