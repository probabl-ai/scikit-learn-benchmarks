#!/bin/bash
# Exp 23: split_indices and find_node_split guards, one at a time, in passive mode with the top band
# capped at 8 threads, on the cases that get several threads. Rebuilds the knobs first, under the lock.
# usage: run_exp23.sh "MAXT|PROFILE|KEY=VAL ..." ...
cd ~/work/hgb-regions
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
if [ -d benchrepo ]; then export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo; BUILD="pixi run bash build_knob.sh"; else export PYTHONPATH=~/scikit-learn-benchmarks; BUILD="bash build_knob.sh"; fi
KEEP=(); [ "$(uname)" = Darwin ] && KEEP=(caffeinate -dims)
PY=venv-c325main/bin/python; [ -x $PY ] || PY=venv-main/bin/python
WORK=N-wide:10000:300,O:300000:100
CASES=L-thin,L,L-stumps,L-leaf255,N-wide,O,kddcup09_churn,covtype,susy,year_prediction_msd
B=c325knob:main~PCAP=8
V=$B,$B~SM=500,$B~SM=2000,$B~SM=20000,$B~SM=100000,$B~FS=0,$B~FS=1,$B~FS=2,$B~FS=100000,c325knob:main
benchlock --wait 14400 --tag "hgb heuristics: rebuild knobs" all -- "${KEEP[@]}" bash -lc "cd ~/work/hgb-regions && $BUILD" > build_knob2.log 2>&1 || { echo "### build failed"; exit 1; }
for spec in "$@"; do
  IFS='|' read -r T prof envs <<< "$spec"
  envs=${envs//\$HOME/$HOME}
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  benchlock --wait 14400 --tag "hgb heuristics: $prof" all -- "${KEEP[@]}" $PY bench.py --profile $prof --threads $T --variants "$V" --cases $CASES --workloads $WORK --n-runs 1 --reps 3 --phases "${eargs[@]}" --out $prof.jsonl
  echo "### $prof rc=$?"
done
echo "### all done"
