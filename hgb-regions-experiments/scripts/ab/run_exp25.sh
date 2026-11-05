#!/bin/bash
# Exp 25: candidate thread stairs on shapes not used for tuning, 5 processes of 1 fit, interleaved.
# Rebuilds the knobs first if needed. Each spec gets one benchlock call.
# usage: run_exp25.sh SLOT "MAXT|PROFILE|VARIANTS|KEY=VAL ..." ...
cd ~/work/hgb-regions
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
if [ -d benchrepo ]; then export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo; BUILD="pixi run bash build_knob.sh"; else export PYTHONPATH=~/scikit-learn-benchmarks; BUILD="bash build_knob.sh"; fi
KEEP=(); [ "$(uname)" = Darwin ] && KEEP=(caffeinate -dims)
PY=venv-c325main/bin/python; [ -x $PY ] || PY=venv-main/bin/python
SLOT=$1; shift
WORK=h-100K:5000:20,h-200K:2000:100,h-500K:100000:5,h-600K:30000:20,h-1.5M:15000:100,h-3M:150000:20,h-3M-l7:150000:20:7,h-10M:500000:20,h-10M-w:50000:200,h-10M-l127:100000:100:127,h-25M:1000000:25,h-40M:200000:200,h-40M-l127:200000:200:127,h-80M:8000000:10,h-200M:2000000:100
CASES=$(echo $WORK | tr ',' '\n' | cut -d: -f1 | paste -sd, -)
if ! grep -q HGB_ASTAIR skl-c325knob/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py; then
  benchlock --wait 14400 --tag "hgb heuristics: rebuild knobs" $SLOT -- "${KEEP[@]}" bash -lc "cd ~/work/hgb-regions && $BUILD" > build_knob3.log 2>&1 || { echo "### build failed"; exit 1; }
fi
for spec in "$@"; do
  IFS='|' read -r T prof vars envs <<< "$spec"
  envs=${envs//\$HOME/$HOME}
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  benchlock --wait 14400 --tag "hgb heuristics: $prof" $SLOT -- "${KEEP[@]}" $PY bench.py --profile $prof --threads $T --variants "$vars" --cases $CASES --workloads $WORK --n-runs 1 --reps ${REPS:-5} --phases "${eargs[@]}" --out $prof.jsonl
  echo "### $prof rc=$?"
done
echo "### all done"
