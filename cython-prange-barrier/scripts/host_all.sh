#!/bin/bash
# Run under benchlock, on the host: builds, then every A/B configuration. usage: host_all.sh JOBS THREADS REPS CONFIG...
# CONFIG is PROFILE or PROFILE,KEY=VAL,KEY=VAL
cd ~/work/cython-prange; jobs=$1; T=$2; R=$3; shift 3
bash build.sh $jobs m331=d5a61d1149cb=331 mA=d5a61d1149cb=fixA mB=d5a61d1149cb=fixB f331=9ade78bf1a25=331 fB=9ade78bf1a25=fixB nb331=10b5afac1433=331 2>&1 | tee build.log
for b in m331 mA mB f331 fB nb331; do [ -x venv-$b/bin/python ] && (cd /tmp && ~/work/cython-prange/venv-$b/bin/python -c 'import sklearn.cluster' ) || { echo "build $b broken"; exit 1; }; done
for cfg in "$@"; do IFS=, read -r -a parts <<< "$cfg"; bash run_ab.sh "${parts[0]}" $T $R "${parts[@]:1}" 2>&1 | tee -a ab.log; done
echo "# all done $(date -u +%H:%M)"
