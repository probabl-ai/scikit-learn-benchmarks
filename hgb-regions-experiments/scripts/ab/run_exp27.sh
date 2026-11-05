#!/bin/bash
# Exp 27: the retuned rule, as real code, on the 15 hgb_scalability cases and on 15 shapes used nowhere
# else, 5 processes of 1 fit, interleaved. Rebuilds retune first. Each spec gets one benchlock call.
# usage: run_exp27.sh SLOT "MAXT|PROFILE|VARIANTS|KEY=VAL ..." ...
cd ~/work/hgb-regions
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
if [ -d benchrepo ]; then export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo; BUILD="pixi run bash build_retune.sh"; else export PYTHONPATH=~/scikit-learn-benchmarks; BUILD="bash build_retune.sh"; fi
KEEP=(); [ "$(uname)" = Darwin ] && KEEP=(caffeinate -dims)
PY=venv-c325main/bin/python; [ -x $PY ] || PY=venv-main/bin/python
SLOT=$1; shift
WORK=t-150K:3000:50,t-400K:20000:20,t-1M:50000:20,t-4M:40000:100,t-6M-l127:120000:50:127,t-8M:200000:40,t-15M:750000:20,t-22M:110000:200,t-30M:1500000:20,t-48M:800000:60,t-60M-l7:300000:200:7,t-70M:350000:200,t-90M:3000000:30,t-150M:500000:300,t-300M:3000000:100
ORIG=XS,S-thin,S,M,M-very-thin,L-thin,L,L-stumps,L-leaf255,amazon_employee_access,ames_housing,covtype,kddcup09_churn,susy,year_prediction_msd
CASES=$ORIG,$(echo $WORK | tr ',' '\n' | cut -d: -f1 | paste -sd, -)
[ -n "$SKIP_BUILD" ] || benchlock --wait 14400 --tag "hgb heuristics: rebuild retune" $SLOT -- "${KEEP[@]}" bash -lc "cd ~/work/hgb-regions && $BUILD" > build_retune2.log 2>&1 || { echo "### build failed"; exit 1; }
[ -n "$SKIP_BUILD" ] || tail -n 2 build_retune2.log
for spec in "$@"; do
  IFS='|' read -r T prof vars envs <<< "$spec"
  envs=${envs//\$HOME/$HOME}
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  S=${SLOT_OVERRIDE:-$SLOT}; [[ "$prof" == *-all ]] && S=all
  benchlock --wait 28800 --tag "hgb heuristics: $prof" $S -- "${KEEP[@]}" $PY bench.py --profile $prof --threads $T --variants "$vars" --cases $CASES --workloads $WORK --n-runs 1 --reps ${REPS:-5} --phases "${eargs[@]}" --out $prof.jsonl
  echo "### $prof rc=$?"
done
echo "### all done"
