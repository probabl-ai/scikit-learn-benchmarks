#!/bin/bash
# Run under benchlock. Build probes/race.pyx with one Cython version per argument and run its two checks.
# usage: race.sh OUTDIR CYTHON_SRC...
HERE=$(cd "$(dirname "$0")/.." && pwd); out=$1; shift; mkdir -p "$out"; cd "$out"
for cysrc in "$@"; do for tc in gcc clang; do
  name=$(basename "$cysrc")-$tc; d=$PWD/$name; mkdir -p "$d"; cp "$HERE/probes/race.pyx" "$d/"
  if [ $tc = gcc ]; then spec="gcc_linux-64=15"; cc=x86_64-conda-linux-gnu-gcc; else spec="clang=21 -s llvm-openmp"; cc=clang; fi
  pixi exec -c conda-forge -s python=3.13 -s setuptools -s $spec -- bash -c "
    cd '$d'; PYTHONPATH='$cysrc' python '$cysrc/cython.py' -3 race.pyx -o race.c &&
    $cc -O2 -fopenmp -fPIC -shared -L\$CONDA_PREFIX/lib -Wl,-rpath,\$CONDA_PREFIX/lib -I\$(python -c 'import sysconfig;print(sysconfig.get_paths()[\"include\"])') race.c -o race\$(python -c 'import sysconfig;print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') &&
    grep -n '#pragma omp' race.c | grep -E 'omp (for|barrier)' | sed -n 1,4p
    objdump -dr race*.so | grep -cE 'GOMP_barrier|GOMP_loop_end|__kmpc_barrier' | sed 's/^/barrier-call sites in .so: /'
    python -I -c \"
import sys; sys.path.insert(0, '$d')
import race
print('$name after_loop  (threads that saw unwritten items, of 3 early ones):', [race.after_loop(2000, 4) for _ in range(10)])
print('$name between_loops (items found unwritten):', [race.between_loops(2000, 4) for _ in range(10)])
\""
done; done
