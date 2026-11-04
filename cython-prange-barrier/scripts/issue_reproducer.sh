#!/bin/bash
# Run under benchlock. Build probes/issue_reproducer.pyx per Cython source and compiler. usage: issue_reproducer.sh OUTDIR CYTHON_SRC...
HERE=$(cd "$(dirname "$0")/.." && pwd); out=$1; shift; mkdir -p "$out"; cd "$out"
for cysrc in "$@"; do for tc in gcc clang; do
  d=$PWD/$(basename "$cysrc")-$tc; mkdir -p "$d"; cp "$HERE/probes/issue_reproducer.pyx" "$d/repro.pyx"
  if [ $tc = gcc ]; then spec="gcc_linux-64=15"; cc=x86_64-conda-linux-gnu-gcc; else spec="clang=21"; cc=clang; fi
  pixi exec -c conda-forge -s python=3.13 -s setuptools -s llvm-openmp -s $spec -- bash -c "
    cd '$d'; PYTHONPATH='$cysrc' python '$cysrc/cython.py' -3 repro.pyx -o repro.c &&
    $cc -O2 -fopenmp -fPIC -shared -L\$CONDA_PREFIX/lib -Wl,-rpath,\$CONDA_PREFIX/lib -I\$CONDA_PREFIX/include -I\$(python -c 'import sysconfig;print(sysconfig.get_paths()[\"include\"])') repro.c -o repro\$(python -c 'import sysconfig;print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') &&
    echo \"\$(basename $d) \$($cc --version | head -1 | cut -c1-50)\" && grep -E '#pragma omp for' repro.c | sed 's/^ *//' &&
    python -I -c \"
import sys; sys.path.insert(0, '$d')
import repro
print([repro.run() for _ in range(10)])\"" 2>&1
done; done
