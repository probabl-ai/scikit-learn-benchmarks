#!/bin/bash
# Exp 17: P-core and full-machine msd, all idle states vs only C1 (PM QoS latency limit of
# 2 us, released when the holder exits). Under benchlock.
cd ~/work/hgb-regions; export PYTHONPATH=~/scikit-learn-benchmarks
OMP=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
V=abs,abs+fuse3,pr+fuse3,dyn:abs+fuse3,nb:abs+fuse3
run() { # label threads cpus env...
  local label=$1 T=$2 cpus=$3; shift 3
  local eargs=(); for kv in "$@"; do eargs+=(--env "$kv"); done
  local c=(); [ -n "$cpus" ] && c=(--cpus $cpus)
  venv-main/bin/python bench.py --threads $T "${c[@]}" --cases year_prediction_msd --variants $V --n-runs 1 --reps 3 --phases --profile exp17-$label "${eargs[@]}" --out exp17-$label.jsonl
}
sets() { # suffix
  run omp-P-$1 8 0-7 LD_PRELOAD=$OMP
  run gomp-P-$1 8 0-7
  run omp-all-$1 24 "" LD_PRELOAD=$OMP
  run gomp-all-$1 24 ""
}
sets deep
sudo python3 -c "
import os, struct, time
fd = os.open('/dev/cpu_dma_latency', os.O_WRONLY)
os.write(fd, struct.pack('i', 2))
time.sleep(1800)" &
QOS=$!
trap 'sudo pkill -P $QOS 2>/dev/null; sudo kill $QOS 2>/dev/null' EXIT
sleep 1
sets c1
