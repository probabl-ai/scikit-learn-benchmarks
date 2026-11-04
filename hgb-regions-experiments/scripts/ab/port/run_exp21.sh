#!/bin/bash
# Exp 21: main, #34935 and hgb/merge-regions on Cython 3.2.5, interleaved, 15 cases, 5 processes
# of 1 fit. usage: run_exp21.sh "THREADS|PROFILE|KEY=VAL ..." ...
cd ~/work/hgb-regions; export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo
for spec in "$@"; do
  IFS='|' read -r T prof envs <<< "$spec"
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  venv-c325main/bin/python bench.py --profile $prof --threads $T --variants c325main:main,c325pr:pr,c325clean:pr --n-runs 1 --reps 5 --phases "${eargs[@]}" --out $prof.jsonl
done
echo "### all done"
