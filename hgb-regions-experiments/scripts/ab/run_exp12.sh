#!/bin/bash
# Exp 12: msd phase times, both runtimes, gamer. Run under benchlock.
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
OMP=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
V=main,pr,abs,abs+fuse,abs+fuse2,abs+fuse3,none+fuse3,pr+fuse3
for r in 1 2 3; do
  venv-main/bin/python bench.py --profile exp12-gomp --threads 24 --cases year_prediction_msd --variants $V --n-runs 1 --phases --out exp12-gomp.jsonl
  venv-main/bin/python bench.py --profile exp12-omp --threads 24 --cases year_prediction_msd --variants $V --n-runs 1 --phases --env LD_PRELOAD=$OMP --out exp12-omp.jsonl
done
