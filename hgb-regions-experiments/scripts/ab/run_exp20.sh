#!/bin/bash
# Exp 20: A/B with Cython 3.2.5 (the pin) and 3.3.x, interleaved, 15 cases, 5 processes of 1 fit.
# Under benchlock. usage: run_exp20.sh THREADS "PROFILE|KEY=VAL ..." ...
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
bash build_cython325.sh || exit 1
T=$1; shift
for spec in "$@"; do
  IFS='|' read -r prof envs <<< "$spec"
  envs=${envs//\$HOME/$HOME}
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  venv-main/bin/python bench.py --profile $prof --threads $T --variants c325pr:pr,c325clean:pr,pr,clean:pr --n-runs 1 --reps 5 --phases "${eargs[@]}" --out $prof.jsonl
  venv-main/bin/python bench.py --profile $prof-main --threads $T --cases year_prediction_msd --variants main,c325main:main --n-runs 1 --reps 3 --phases "${eargs[@]}" --out $prof-main.jsonl
done
echo "### all done"
