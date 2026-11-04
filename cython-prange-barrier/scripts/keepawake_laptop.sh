#!/bin/bash
# Run under benchlock on intel-laptop: same idle table as intel-gamer-gpu, but Fix B wins with libomp on main.
# Does it reach deep states as often as gamer, and do the timers change anything?
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
L=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-mkl/lib/libomp.so
test -f "$L" || { echo "missing $L"; exit 1; }
out=$PWD/placement/keepawake_laptop
mkdir -p "$out"
gcc -O2 -pthread placement/keepawake.c -o placement/keepawake_helper || exit 1
helper=

stop_helper() {
  if [ -n "$helper" ]; then
    kill -TERM "$helper"
    wait "$helper"
    helper=
  fi
}
trap stop_helper EXIT

snap() {
  grep -H . /sys/devices/system/cpu/cpu*/cpuidle/state*/usage /sys/devices/system/cpu/cpu*/cpuidle/state*/time /sys/devices/system/cpu/cpu*/cpuidle/state*/name > "$1"
}

for rep in 1 2 3 4 5; do
  for period in 0 200 5000; do
    if [ "$period" != 0 ]; then
      placement/keepawake_helper "$period" 16 &
      helper=$!
      sleep 0.5
    fi
    for b in m331 mB2; do
      snap "$out/t$period-$b-$rep.before"
      venv-m331/bin/python bench_hgb.py --profile "t$period" --threads 16 --cpus 0-15 \
        --cases year_prediction_msd --variants "$b:main" --n-runs 1 --reps 1 \
        --env "LD_PRELOAD=$L" --out "$out/t$period.jsonl" | grep rc=
      snap "$out/t$period-$b-$rep.after"
    done
    stop_helper
  done
done
echo "# done $(date -u +%H:%M)"
