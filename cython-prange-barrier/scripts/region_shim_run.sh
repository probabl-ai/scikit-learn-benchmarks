#!/bin/bash
# Run under benchlock: one fit of year_prediction_msd per build and rep, alternating, with probes/region_shim.c.
# usage: region_shim_run.sh NAME THREADS CPUS REPS LIBOMP(or none)
cd ~/work/cython-prange
export PYTHONPATH=~/scikit-learn-benchmarks
name=$1; T=$2; C=$3; R=$4; L=$5
mkdir -p placement/shim
gcc -O2 -fPIC -shared placement/region_shim.c -o placement/region_shim.so -ldl || exit 1
preload=$PWD/placement/region_shim.so
[ "$L" = none ] || preload=$preload:$L
for rep in $(seq 1 "$R"); do
  for b in m331 mB2; do
    out=$PWD/placement/shim/$name-$b-$rep.txt
    rm -f "$out"
    venv-m331/bin/python bench_hgb.py --profile "$name" --threads "$T" --cpus "$C" \
      --cases year_prediction_msd --variants "$b:main" --n-runs 1 --reps 1 \
      --env "LD_PRELOAD=$preload" --env "SHIM_OUT=$out" --out "placement/shim/$name.jsonl" | grep rc=
  done
done
echo "# done $name $(date -u +%H:%M)"
