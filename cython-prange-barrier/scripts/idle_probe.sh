#!/bin/bash
# Run under benchlock on intel-gamer-gpu: is Fix B's wake-up delay with libomp caused by CPU idle states?
# Records idle state usage around each fit, then repeats with a SCHED_IDLE spinner on each CPU, which keeps
# cores out of idle states and yields at once to a waking thread. Also the toy with HGB-sized regions.
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
L=$HOME/scikit-learn-benchmarks/.pixi/envs/sklearn-dev-libomp/lib/libomp.so
test -f "$L" || { echo "missing $L"; exit 1; }
out=$PWD/placement/idle
mkdir -p "$out"
gcc -O2 -fPIC -shared placement/region_shim.c -o placement/region_shim.so -ldl || exit 1
gcc -O1 -fopenmp placement/barrier_placement.c -o placement/barrier_placement || exit 1
preload=$PWD/placement/region_shim.so:$L
spinners=()

stop_spinners() {
  [ ${#spinners[@]} -gt 0 ] && kill "${spinners[@]}" 2>/dev/null
  wait 2>/dev/null
  spinners=()
}
trap stop_spinners EXIT

snap() {
  grep -H . /sys/devices/system/cpu/cpu*/cpuidle/state*/usage /sys/devices/system/cpu/cpu*/cpuidle/state*/time /sys/devices/system/cpu/cpu*/cpuidle/state*/name > "$1"
}

fits() {  # NAME
  for rep in 1 2 3 4 5; do
    for b in m331 mB2; do
      snap "$out/$1-$b-$rep.before"
      rm -f "$out/$1-$b-$rep.txt"
      venv-m331/bin/python bench_hgb.py --profile "$1" --threads 24 --cpus 0-23 \
        --cases year_prediction_msd --variants "$b:main" --n-runs 1 --reps 1 \
        --env "LD_PRELOAD=$preload" --env "SHIM_OUT=$out/$1-$b-$rep.txt" --out "$out/$1.jsonl" | grep rc=
      snap "$out/$1-$b-$rep.after"
    done
  done
}

toy() {  # NAME
  for rep in 1 2 3 4 5; do
    for b in 0 1; do
      echo "cfg=$1 rt=libomp rep=$rep $(LD_PRELOAD=$L placement/barrier_placement $b 24 2000 12000 50000)"
    done
  done
}

toy base
fits base
for c in $(seq 0 23); do
  chrt -i 0 taskset -c "$c" sh -c 'while :; do :; done' &
  spinners+=($!)
done
echo "# spinners: ${#spinners[@]}"
toy spin
fits spin
stop_spinners
toy base2
echo "# done $(date -u +%H:%M)"
