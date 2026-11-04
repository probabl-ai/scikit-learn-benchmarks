#!/bin/bash
# Run under benchlock on intel-gamer-gpu: follow-up to keepawake_probe.sh, whose 5 ms arm wasn't a neutral control
# since it also removed C3. Longer timer periods look for a helper that leaves deep states on, and for whether
# Fix B's penalty comes back along with them.
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
L=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
test -f "$L" || { echo "missing $L"; exit 1; }
out=$PWD/placement/keepawake_dose
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
  for period in 0 5000 20000 50000 200000; do
    if [ "$period" != 0 ]; then
      placement/keepawake_helper "$period" 24 &
      helper=$!
      sleep 0.5
    fi
    for b in m331 mB2; do
      snap "$out/t$period-$b-$rep.before"
      venv-m331/bin/python bench_hgb.py --profile "t$period" --threads 24 --cpus 0-23 \
        --cases year_prediction_msd --variants "$b:main" --n-runs 1 --reps 1 \
        --env "LD_PRELOAD=$L" --out "$out/t$period.jsonl" | grep rc=
      snap "$out/t$period-$b-$rep.after"
    done
    stop_helper
  done
done
echo "# done $(date -u +%H:%M)"
