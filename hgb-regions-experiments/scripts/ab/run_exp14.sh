#!/bin/bash
# Exp 14: why fused variants lose to abs on msd with libomp (gamer). Under benchlock.
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
OMP=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
bash build_branch.sh nb hgb/fewer-regions-nb
V=main,abs,abs+fuse,abs+fuse3,pr+fuse3,nb:abs,nb:abs+fuse3,nb:pr+fuse3
B="venv-main/bin/python bench.py --threads 24 --cases year_prediction_msd --variants $V --n-runs 1 --reps 3 --phases"
$B --profile exp14-omp --env LD_PRELOAD=$OMP --out exp14-omp.jsonl
$B --profile exp14-gomp --out exp14-gomp.jsonl
$B --profile exp14-omp-spin --env LD_PRELOAD=$OMP --env KMP_BLOCKTIME=200ms --env SKLEARN_OMP_WAIT_POLICY=passive --out exp14-omp-spin.jsonl
venv-main/bin/python bench.py --threads 16 --cpus 8-23 --cases year_prediction_msd --variants $V --n-runs 1 --reps 3 --phases --profile exp14-omp-Ecores --env LD_PRELOAD=$OMP --out exp14-omp-Ecores.jsonl
venv-main/bin/python bench.py --threads 16 --cpus 8-23 --cases year_prediction_msd --variants $V --n-runs 1 --reps 3 --phases --profile exp14-gomp-Ecores --out exp14-gomp-Ecores.jsonl
