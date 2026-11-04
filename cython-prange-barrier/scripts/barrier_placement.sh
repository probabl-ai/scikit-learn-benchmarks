#!/bin/bash
# Run under benchlock: probes/barrier_placement.c with and without the barrier, interleaved.
# usage: barrier_placement.sh THREADS LIBOMP SOURCE
T=$1; L=$2; src=$3
test -f "$L" || { echo "missing $L"; exit 1; }
bin=$(mktemp -d)/probe
gcc -O1 -fopenmp "$src" -o "$bin" || exit 1
gcc --version | head -1
for rep in 1 2 3 4 5; do
  for n in 100000 1000000; do
    regions=$((n == 100000 ? 4000 : 1000))
    for b in 0 1; do
      echo "rt=libgomp rep=$rep $("$bin" $b "$T" $n 200000 $regions)"
      echo "rt=libomp rep=$rep $(LD_PRELOAD=$L "$bin" $b "$T" $n 200000 $regions)"
    done
  done
done
