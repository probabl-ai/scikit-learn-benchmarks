#!/bin/bash
# Fetches Intel's OpenMP runtime (libiomp5.so) from the intel-openmp wheel into
# vendor/intel-openmp, to compare with conda-forge's LLVM libomp under MKL.
set -euo pipefail
cd "$(dirname "$0")/.."
dest=vendor/intel-openmp
mkdir -p "$dest/wheel"
pixi exec -s pip -- pip download --no-deps --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --python-version 3.12 \
  -d "$dest/wheel" "intel-openmp==2026.1.*"
whl=$(ls -t "$dest"/wheel/intel_openmp-2026.1*.whl | head -1)
pixi run python -I -c 'import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
n = [i for i in z.namelist() if i.endswith("/libiomp5.so")][0]
open(sys.argv[2], "wb").write(z.read(n))' "$whl" "$dest/libiomp5.so"
basename "$whl" > "$dest/version.txt"
