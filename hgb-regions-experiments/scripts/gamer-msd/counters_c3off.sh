#!/bin/bash
# One PR fit on year_prediction_msd (23 threads), serial vs parallel find_split, with OS counters.
cd ~/scikit-learn-benchmarks
D=~/work/gamer-run/msd
c3() { cat /sys/devices/system/cpu/cpu*/cpuidle/state3/usage | paste -sd+ | bc; }; c2() { cat /sys/devices/system/cpu/cpu*/cpuidle/state2/usage | paste -sd+ | bc; }
for rep in 1 2; do
for pol in default active; do
  extra=""; [ $pol = active ] && extra="SKLEARN_OMP_WAIT_POLICY=active"
  b=$(c3); b2=$(c2)
  /usr/bin/time -f "wall=%e user=%U sys=%S vol_cs=%w invol_cs=%c" env OMP_NUM_THREADS=24 $extra \
    pixi run --frozen -e sklearn-dev python $D/patched_runner.py --case-file $D/case.json --n-runs 3 --output-jsonl $D/c.jsonl 2> $D/c.err
  a=$(c3); a2=$(c2)
  echo "policy=$pol C3_entries=$((a-b)) C2_entries=$((a2-b2)) $(tail -n 1 $D/c.err) fits=$(python3 -c "import json;print([round(json.loads(l)['time_ms']['fit']/1000,1) for l in open('$D/c.jsonl')])")"
  rm -f $D/c.jsonl
done
done
