#!/bin/bash
# Build an OpenBLAS checkout with the conda-forge linux-64 flags (pthreads).
# usage: scripts/build_openblas.sh SRC PREFIX [extra make flags...]
# e.g.   scripts/build_openblas.sh openblas/src builds/base
#        scripts/build_openblas.sh openblas/src builds/t256 NUM_THREADS=256
# Later flags override the defaults (make keeps the last assignment).
set -euo pipefail
SRC=$(realpath "$1")
mkdir -p "$2"
PREFIX=$(realpath "$2")
shift 2
JOBS=${JOBS:-32}
cd "$SRC"
make clean >/dev/null 2>&1 || true
FLAGS="DYNAMIC_ARCH=1 BINARY=64 NO_LAPACK=0 TARGET=PRESCOTT BUILD_BFLOAT16=1 NO_AFFINITY=1 USE_THREAD=1 NUM_THREADS=128 USE_OPENMP=0 INTERFACE64=0 $*"
pixi exec -s gcc_linux-64=15 -s gfortran_linux-64=15 -s make -- bash -c "
set -eu
export FFLAGS=\"\${FFLAGS:-} -frecursive\"
make -j$JOBS $FLAGS CC=\$CC FC=\$FC HOSTCC=\$CC >$PREFIX/build.log 2>&1
make $FLAGS CC=\$CC FC=\$FC HOSTCC=\$CC PREFIX=$PREFIX install >>$PREFIX/build.log 2>&1
"
git -C "$SRC" rev-parse HEAD > "$PREFIX/commit.txt"
git -C "$SRC" diff > "$PREFIX/patch.diff"
echo "$FLAGS" > "$PREFIX/flags.txt"
ls "$PREFIX/lib"/libopenblas*.so*
