#!/bin/bash
# A/B: only POLL+C1 allowed vs all idle states, PR fit (serial vs parallel find_split) + wake2.
# Restores the starting state (C3 disabled) on exit.
D=~/work/gamer-run/msd
W=~/work/gamer-run/wake
restore() { sudo cpupower idle-set -E >/dev/null; sudo cpupower idle-set -d 3 >/dev/null; echo "restored: $(cat /sys/devices/system/cpu/cpu0/cpuidle/state*/disable | paste -sd' ')"; }
trap restore EXIT
usage() { cat /sys/devices/system/cpu/cpu*/cpuidle/state$1/usage | paste -sd+ | bc; }
for rep in 1 2; do
for st in c1only all; do
  sudo cpupower idle-set -E >/dev/null
  [ $st = c1only ] && sudo cpupower idle-set -D 2 >/dev/null
  echo "### rep=$rep states=$st disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state*/disable | paste -sd' ')"
  echo "== wake2 passive 23T"; OMP_NUM_THREADS=23 $W/wake2 1000 32 4 0 160 500 2000
  cd ~/scikit-learn-benchmarks
  for pol in default active; do
    extra=""; [ $pol = active ] && extra="SKLEARN_OMP_WAIT_POLICY=active"
    b1=$(usage 1); b2=$(usage 2); b3=$(usage 3)
    /usr/bin/time -f "wall=%e user=%U sys=%S vol_cs=%w invol_cs=%c" env OMP_NUM_THREADS=24 $extra \
      pixi run --frozen -e sklearn-dev python $D/patched_runner.py --case-file $D/case.json --n-runs 3 --output-jsonl $D/c.jsonl 2> $D/c.err
    echo "policy=$pol C1=$(( $(usage 1)-b1 )) C2=$(( $(usage 2)-b2 )) C3=$(( $(usage 3)-b3 )) $(tail -n 1 $D/c.err) fits=$(python3 -c "import json;print([round(json.loads(l)['time_ms']['fit']/1000,1) for l in open('$D/c.jsonl')])")"
    rm -f $D/c.jsonl
  done
done
done
echo "== done $(date -Is)"
