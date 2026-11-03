#!/bin/bash
cd ~/scikit-learn-benchmarks
W=~/work/hgb-regions
for v in pr abs+fuse3 pr abs+fuse3; do
  X="SKLEARN_HGB_GUARD=${v%%+*}"; [[ $v == *fuse3 ]] && X="$X SKLEARN_HGB_EXP=fuse3"
  echo "== $v"
  sudo env $X OMP_NUM_THREADS=24 PYTHONPATH=~/scikit-learn-benchmarks OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=1 KMP_BLOCKTIME=0 \
    turbostat -q -S -- sudo -u alacote env $X OMP_NUM_THREADS=24 PYTHONPATH=~/scikit-learn-benchmarks $W/venv-exp/bin/python -m sklbench.runners.estimator \
      --case-file ~/work/gamer-run/msd/case.json --n-runs 1 --output-jsonl /tmp/turbo_$$.jsonl 2>&1 | grep -vE "^(INFO|WARNING|DEBUG)" | tail -4
  python3 -c "import json;print(\"fit\", [round(json.loads(l)[\"time_ms\"][\"fit\"]/1000,2) for l in open(\"/tmp/turbo_$$.jsonl\")])"; rm -f /tmp/turbo_$$.jsonl
done
