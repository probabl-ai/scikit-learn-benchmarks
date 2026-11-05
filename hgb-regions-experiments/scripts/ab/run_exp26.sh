#!/bin/bash
# Exp 26: where the passive cap goes from 8 to 16, on shapes used nowhere else, 5 processes of 1 fit.
# SLOT, WORK and VARS can come from the environment. usage: run_exp26.sh "MAXT|PROFILE|KEY=VAL ..." ...
cd ~/work/hgb-regions
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
if [ -d benchrepo ]; then export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo; else export PYTHONPATH=~/scikit-learn-benchmarks; fi
PY=venv-c325main/bin/python; [ -x $PY ] || PY=venv-main/bin/python
WORK=${WORK:-b-30M:600000:50,b-36M:120000:300,b-45M:1500000:30,b-50M:250000:200,b-60M:600000:100,b-60M-t:4000000:15,b-60M-w:120000:500,b-60M-l127:400000:150:127,b-75M:150000:500,b-100M:1000000:100,b-120M:400000:300}
CASES=$(echo $WORK | tr ',' '\n' | cut -d: -f1 | paste -sd, -)
BASE=c325knob:main~PSTAIR=2000000:1/20000000:4/999999999999
VARS=${VARS:-c325knob:main,$BASE:8,$BASE:16}
for spec in "$@"; do
  IFS='|' read -r T prof envs <<< "$spec"
  envs=${envs//\$HOME/$HOME}
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  benchlock --wait 14400 --tag "hgb heuristics: $prof" ${SLOT:-all} -- $PY bench.py --profile $prof --threads $T --variants "$VARS" --cases $CASES --workloads $WORK --n-runs 1 --reps ${REPS:-5} --phases "${eargs[@]}" --out $prof.jsonl
  echo "### $prof rc=$?"
done
echo "### all done"
