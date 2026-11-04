#!/bin/bash
# One fit of a synthetic case and a real one per build, to check the setup before a long run.
cd ~/work/hgb-regions; export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
export SKLBENCH_REPO=$PWD/benchrepo PYTHONPATH=$PWD/benchrepo
rm -f smoke.jsonl
venv-c325main/bin/python bench.py --profile smoke --threads $1 --cases XS,ames_housing --variants c325main:main,c325pr:pr,c325clean:pr --n-runs 1 --reps 1 --phases --out smoke.jsonl
venv-c325main/bin/python -c "
import json
for l in open('smoke.jsonl'):
    r = json.loads(l)
    if r['rc'] != 0: print(r['case'], r['variant'], r['stderr'][-1500:])
"
echo "### smoke done"
