#!/bin/bash
# Exp 24: guards in active mode, one at a time, at a provisional stair of thread caps. Rebuilds the
# knobs first, under the lock. usage: run_exp24.sh SLOT "MAXT|PROFILE|KEY=VAL ..." ...
cd ~/work/hgb-regions
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
if [ -d benchrepo ]; then export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo; BUILD="pixi run bash build_knob.sh"; else export PYTHONPATH=~/scikit-learn-benchmarks; BUILD="bash build_knob.sh"; fi
KEEP=(); [ "$(uname)" = Darwin ] && KEEP=(caffeinate -dims)
PY=venv-c325main/bin/python; [ -x $PY ] || PY=venv-main/bin/python
SLOT=$1; shift
WORK=N-thin:200000:10,N:40000:50,N-wide:10000:300,O:300000:100
CASES=S-thin,S,M,M-very-thin,L-thin,L,L-stumps,L-leaf255,N-thin,N,N-wide,O,ames_housing,kddcup09_churn,amazon_employee_access,covtype,susy,year_prediction_msd
B=c325knob:main~ASTAIR=200000:4/5000000:8/20000000:16
V=$B,$B~SM=100,$B~SM=2000,$B~SM=5000,$B~SM=20000,$B~FS=1,$B~FS=2,$B~FS=5,c325knob:main
benchlock --wait 14400 --tag "hgb heuristics: rebuild knobs" $SLOT -- "${KEEP[@]}" bash -lc "cd ~/work/hgb-regions && $BUILD" > build_knob3.log 2>&1 || { echo "### build failed"; exit 1; }
for spec in "$@"; do
  IFS='|' read -r T prof envs <<< "$spec"
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  benchlock --wait 14400 --tag "hgb heuristics: $prof" $SLOT -- "${KEEP[@]}" $PY bench.py --profile $prof --threads $T --variants "$V" --cases $CASES --workloads $WORK --n-runs 1 --reps 3 --phases "${eargs[@]}" --out $prof.jsonl
  echo "### $prof rc=$?"
done
echo "### all done"
