#!/bin/bash
# Exp 15: dynamic schedule of the merged feature loop, msd, gamer. Under benchlock.
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
OMP=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
bash build_branch.sh dyn hgb/fewer-regions-dyn
V=abs,pr+fuse3,abs+fuse3,dyn:pr+fuse3,dyn:abs+fuse3,nb:pr+fuse3
B="venv-main/bin/python bench.py --cases year_prediction_msd --variants $V --n-runs 1 --reps 3 --phases"
$B --threads 24 --profile exp15-omp --env LD_PRELOAD=$OMP --out exp15-omp.jsonl
$B --threads 24 --profile exp15-gomp --out exp15-gomp.jsonl
$B --threads 8 --cpus 0-7 --profile exp15-omp-Pcores --env LD_PRELOAD=$OMP --out exp15-omp-Pcores.jsonl
