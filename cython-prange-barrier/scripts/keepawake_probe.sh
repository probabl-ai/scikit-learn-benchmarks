#!/bin/bash
# Run under benchlock on intel-gamer-gpu: does Fix B's slowdown with libomp go away when deep idle states are avoided?
# Interleaves no helper, a 200 us timer on each CPU (below C2's 381 us target residency) and a 5 ms timer (above
# C3's 3144 us, the control for the helper itself). Idle counters around each fit check the manipulation worked.
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
L=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
test -f "$L" || { echo "missing $L"; exit 1; }
out=$PWD/placement/keepawake
mkdir -p "$out"
gcc -O2 -fPIC -shared placement/region_shim.c -o placement/region_shim.so -ldl || exit 1
gcc -O1 -fopenmp placement/barrier_placement.c -o placement/barrier_placement || exit 1
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

start_helper() {  # PERIOD_US, or 0 for no helper
  [ "$1" = 0 ] && return
  placement/keepawake_helper "$1" 24 &
  helper=$!
  sleep 0.5
}

snap() {
  grep -H . /sys/devices/system/cpu/cpu*/cpuidle/state*/usage /sys/devices/system/cpu/cpu*/cpuidle/state*/time /sys/devices/system/cpu/cpu*/cpuidle/state*/name > "$1"
}

fit() {  # NAME BUILD REP PRELOAD
  snap "$out/$1-$2-$3.before"
  venv-m331/bin/python bench_hgb.py --profile "$1" --threads 24 --cpus 0-23 \
    --cases year_prediction_msd --variants "$2:main" --n-runs 1 --reps 1 \
    --env "LD_PRELOAD=$4" --env "SHIM_OUT=$out/$1-$2-$3.txt" --out "$out/$1.jsonl" | grep rc=
  snap "$out/$1-$2-$3.after"
}

for rep in 1 2 3 4 5; do
  for period in 0 200 5000; do
    start_helper "$period"
    for b in m331 mB2; do
      fit "t$period" "$b" "$rep" "$L"
      fit "t${period}_shim" "$b" "$rep" "$PWD/placement/region_shim.so:$L"
    done
    stop_helper
  done
done

# The toy doesn't reproduce the slowdown: does it reach deep states at all?
for rep in 1 2 3; do
  for b in 0 1; do
    snap "$out/toy-b$b-$rep.before"
    echo "cfg=toy rt=libomp rep=$rep $(LD_PRELOAD=$L placement/barrier_placement "$b" 24 2000 12000 50000)"
    snap "$out/toy-b$b-$rep.after"
  done
done
echo "# done $(date -u +%H:%M)"
