#!/bin/bash
# Exp 22: fit time against the thread count, on #34935 plus knobs, Cython 3.2.5. Each profile gets
# one benchlock call. FSVALS adds `~FS=x` at each T > 1 (only useful in passive mode).
# usage: run_exp22.sh SLOT "MAXT|PROFILE|TLIST|FSVALS|KEY=VAL ..." ...
cd ~/work/hgb-regions
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
if [ -d benchrepo ]; then export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo; else export PYTHONPATH=~/scikit-learn-benchmarks; fi
KEEP=(); [ "$(uname)" = Darwin ] && KEEP=(caffeinate -dims)
PY=venv-c325main/bin/python; [ -x $PY ] || PY=venv-main/bin/python
SLOT=$1; shift
WORK=N-thin:200000:10,N:40000:50,N-wide:10000:300,O:300000:100
for spec in "$@"; do
  IFS='|' read -r T prof tlist fsvals envs <<< "$spec"
  envs=${envs//\$HOME/$HOME}
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  v=c325main:main,c325knob:main
  for t in $tlist; do
    v=$v,c325knob:main~T=$t
    if [ "$t" != 1 ]; then for f in $fsvals; do v=$v,c325knob:main~T=$t~FS=$f; done; fi
  done
  for f in $fsvals; do v=$v,c325knob:main~FS=$f; done
  benchlock --wait 7200 --tag "hgb heuristics: $prof" $SLOT -- "${KEEP[@]}" $PY bench.py --profile $prof --threads $T --variants "$v" --workloads $WORK --n-runs 1 --reps ${REPS:-3} --phases "${eargs[@]}" --out $prof.jsonl
  echo "### $prof rc=$?"
done
echo "### all done"
