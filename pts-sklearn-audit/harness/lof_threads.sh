#!/bin/bash
# bench_lof.py segfaults in the PTS sweep on one socket (172 threads): run it at
# fewer OpenMP threads to see if the crash depends on the thread count.
W=$(cd "$(dirname "$0")" && pwd)
cd "$W/home/.phoronix-test-suite/installed-tests/pts/scikit-learn-2.0.0/scikit-learn-1.2.2/benchmarks"
for t in default 64 32; do
    if [ "$t" = default ]; then unset OMP_NUM_THREADS; else export OMP_NUM_THREADS=$t; fi
    start=$(date +%s)
    HOME=$W/home PYTHONPATH=$W/sitecustom timeout 900 "$W/.pixi/envs/default/bin/python3" bench_lof.py > "$W/lof_$t.log" 2>&1
    rc=$?
    echo "OMP_NUM_THREADS=$t rc=$rc $(( $(date +%s) - start ))s"
done
