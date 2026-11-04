#!/bin/bash
# Run under benchlock, on the host: builds, then every A/B configuration. usage: host_all.sh JOBS THREADS REPS CONFIG...
# CONFIG is PROFILE or PROFILE,KEY=VAL,KEY=VAL
cd ~/work/cython-prange
jobs=$1; T=$2; R=$3; shift 3
main=d5a61d1149cb; fused=9ade78bf1a25; manual=10b5afac1433
bash build.sh $jobs m324=$main=324 m331=$main=331 mA=$main=fixA mB2=$main=fixB2 f331=$fused=331 fB2=$fused=fixB2 nb331=$manual=331 2>&1 | tee build2.log
for b in m324 m331 mA mB2 f331 fB2 nb331; do
  (cd /tmp && ~/work/cython-prange/venv-$b/bin/python -c 'import sklearn.cluster') || { echo "build $b broken"; exit 1; }
done
for cfg in "$@"; do
  IFS=, read -r -a parts <<< "$cfg"
  bash run_ab.sh "${parts[0]}" $T $R "${parts[@]:1}" 2>&1 | tee -a ab2.log
done
echo "# all done $(date -u +%H:%M)"
