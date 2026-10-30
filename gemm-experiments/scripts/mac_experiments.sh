#!/bin/bash
# The mac-m2 experiments, by name (see LOG-mac.md). Run on the Mac from the
# repository root, e.g. `ssh -f mac-m2 'cd ~/work/gemm-experiments &&
# nohup scripts/mac_experiments.sh e3_accel_controls >/dev/null 2>&1 &'`.
set -eu
cd "$(dirname "$0")/.."
S=scripts/mac_sweep.sh
SKL="100x100x1000000:NT 10x1000000x100:NN 8x1000000x100:TN 256x100000x100:TN 20x100000x1000:NN 100x200x784:NN"
case $1 in
e3_accel_controls)
  # VECLIB_MAXIMUM_THREADS vs BLASSetThreading vs the $NEWLAPACK symbols.
  $S e3_accel_controls --threads 1,8 --dtype d,s --rounds 3 \
    --variant accelerate \
    --variant 'acc-single:accelerate,GEMMBENCH_BLAS_THREADING=1' \
    --variant 'acc-newlapack:accelerate,GEMMBENCH_SUFFIX=$NEWLAPACK' \
    --shapes 64x64x64 256x256x256 1024x1024x1024 2048x2048x2048 4096x4096x4096 \
      $SKL syrk:100x1000000:N --note "Accelerate threading controls and NEWLAPACK"
  ;;
e3b_accel_background)
  # Accelerate and OpenBLAS confined to the E cores (background QoS).
  PREFIX_CMD="taskpolicy -b" $S e3b_accel_background --threads 1,4 --dtype d,s --rounds 2 \
    --variant accelerate --variant openblas \
    --shapes 256x256x256 1024x1024x1024 $SKL --note "taskpolicy -b: E cores only"
  ;;
e4_callers)
  CMD="bash scripts/mac_callers.sh" $S e4_callers 2 1,2,3,4,5,6,8 d,s 2 \
    512x512x512 2048x2048x2048
  ;;
e5_small)
  $S e5_small --threads 1,4,8 --dtype d,s --rounds 3 \
    --variant accelerate --variant openblas --variant blis \
    --shapes 8x8x8 16x16x16 24x24x24 32x32x32 48x48x48 64x64x64 96x96x96 \
      100x100x100 128x128x128 192x192x192 256x256x256 384x384x384 \
      64x64x64:TN 100x100x100:NT --note "small matrices"
  ;;
e6_numpy)
  CMD="python scripts/mac_numpy_bench.py" $S e6_numpy \
    --venv accel=venvs.noindex/accel --venv openblas=venvs.noindex/openblas \
    --env 'default=' --env 't4=OPENBLAS_NUM_THREADS=4,VECLIB_MAXIMUM_THREADS=4' \
    --env 't1=OPENBLAS_NUM_THREADS=1,VECLIB_MAXIMUM_THREADS=1' --rounds 3
  ;;
e7_thermal)
  # Sustained load: does throughput drift over ~10 minutes?
  $S e7_thermal --threads 4 --dtype s,d --rounds 12 --min-time 5 \
    --variant accelerate --variant openblas \
    --shapes 2048x2048x2048 --note "thermal drift, 12 rounds of 5 s per point"
  ;;
*) echo "unknown experiment $1"; exit 2 ;;
esac
