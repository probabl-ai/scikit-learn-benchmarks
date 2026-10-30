#!/bin/bash
# Experiment 11 (Xeon socket, unpinned like a user's process): is MKL's
# collapse at high thread counts thread migration, or conda-forge's LLVM
# libomp? Binding vs KMP_* settings vs Intel's libiomp5 vs GNU libgomp, and
# whether OMP_PROC_BIND still works when set after the library is loaded
# (what a threadpoolctl feature would have to do).
# usage: scripts/run_mkl_runtime.sh MACHINE THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=$1
THREADS=$2
IOMP=$PWD/vendor/intel-openmp/libiomp5.so
test -f "$IOMP" || scripts/get_intel_openmp.sh
BIND="OMP_PROC_BIND=close,OMP_PLACES=cores"
LATE="GEMMBENCH_LATE_ENV=OMP_PROC_BIND=close;OMP_PLACES=cores"
SHAPES=(256x256x256 1024x1024x1024 4096x4096x4096 7x116203x100:NN
        100x100x1000000:NT 100x200000x100:NN)
pixi run python scripts/sweep.py --out results/$MACHINE/mkl_runtime.csv \
  --threads "$THREADS" --variant openblas --variant mkl \
  --variant "mkl-close:mkl,$BIND" \
  --variant "mkl-bt0:mkl,KMP_BLOCKTIME=0" \
  --variant "mkl-btinf:mkl,KMP_BLOCKTIME=infinite" \
  --variant "mkl-1t:mkl,KMP_HW_SUBSET=1t" \
  --variant "mkl-iomp:mkl,LD_PRELOAD=$IOMP" \
  --variant "mkl-iomp-close:mkl,LD_PRELOAD=$IOMP,$BIND" \
  --variant "mkl-gnu:mkl,MKL_THREADING_LAYER=GNU" \
  --variant "mkl-gnu-close:mkl,MKL_THREADING_LAYER=GNU,$BIND" \
  --variant "mkl-late-dlopen:mkl,$LATE,GEMMBENCH_LATE_ENV_AT=dlopen" \
  --variant "mkl-late-setter:mkl,$LATE,GEMMBENCH_LATE_ENV_AT=setter" \
  --dtype d --rounds 2 --min-time 0.3 \
  --note "exp11 MKL OpenMP runtime, binding and late-binding variants" \
  --shapes "${SHAPES[@]}"
