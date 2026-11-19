#!/bin/bash
# Mixed and staggered concurrent callers (mixbench), base vs sk vs v6.
# usage: [ROUNDS=3] [VARIANTS="base sk v6"] [THREADS="16 86"] [PFX=cc-] run_mix.sh OUT.csv
set -uo pipefail
OUT=$1; PFX=${PFX-cc-}
rep() { local n=$1 s=$2 o=""; for i in $(seq $n); do o+="$s "; done; echo $o; }
declare -A SC
SC[big1_small1]="g:1000:1000:2000 s:100:20000"
SC[big4_small4]="$(rep 4 g:1000:1000:2000) $(rep 4 s:100:20000)"
SC[big1_small7]="g:1000:1000:2000 $(rep 7 s:100:20000)"
SC[big1_gsmall7]="g:1000:1000:2000 $(rep 7 g:64:64:20000)"
SC[ramp8_small]="s:100:20000:0 s:100:20000:0.2 s:100:20000:0.4 s:100:20000:0.6 s:100:20000:0.8 s:100:20000:1.0 s:100:20000:1.2 s:100:20000:1.4"
SC[g200x8]="$(rep 8 g:200:200:50000)"
SC[s200x8]="$(rep 8 s:200:100000)"
SC[s100x8]="$(rep 8 s:100:20000)"
[ -f $OUT ] || echo "label,caller,spec,calls,calls_per_s,mean_ms,scenario,window,T,round" > $OUT
for round in $(seq 1 ${ROUNDS:-3}); do
  for T in ${THREADS:-16 86}; do
    cpus=$(pixi run python -c "import sys; sys.path.insert(0, 'scripts'); import sweep; print(','.join(map(str, sweep.compact_cpus($T))))" | tail -1)
    for sc in "${!SC[@]}"; do
      for W in ${WINDOWS:-0.5 4}; do
        for v in ${VARIANTS:-base sk v6}; do
          env OPENBLAS_NUM_THREADS=$T MIXBENCH_LABEL=$v taskset -c $cpus build/mixbench \
            builds/$PFX$v/lib/libopenblas.so.0 $W ${SC[$sc]} | sed "s/$/,$sc,$W,$T,$round/" >> $OUT
        done
      done
    done
  done
done
echo RUN_DONE
