#!/bin/bash
# Run under benchlock. Warnings of probes/control.c per compiler. usage: control.sh OUTDIR
HERE=$(cd "$(dirname "$0")/.." && pwd); out=$1; mkdir -p "$out"; cd "$out"
for cc in "gcc_linux-64=7 x86_64-conda-linux-gnu-gcc" "gcc_linux-64=8 x86_64-conda-linux-gnu-gcc" "gcc_linux-64=11 x86_64-conda-linux-gnu-gcc" "gcc_linux-64=15 x86_64-conda-linux-gnu-gcc" "clang=21 clang"; do
  spec=${cc% *}; bin=${cc#* }; echo "== $spec"
  pixi exec -c conda-forge -s "$spec" -- $bin -O2 -Wall -Wextra -fopenmp -c "$HERE/probes/control.c" -o /dev/null 2>&1 | grep -E 'warning|In function' | sed 's/.*control.c://'
done
