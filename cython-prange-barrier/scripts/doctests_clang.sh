#!/bin/bash
# Run under benchlock. runtests.py doesn't enable OpenMP on clang, so build test modules by hand and run their doctests.
# usage: doctests_clang.sh CYTHON_SRC OUTDIR TESTNAME...   (each test gets a hang timeout of 90 s, exit 124)
src=$1; out=$2; shift 2; mkdir -p "$out"
for t in "$@"; do
  d=$out/$t; mkdir -p "$d"
  pixi exec -c conda-forge -s python=3.13 -s setuptools -s clang=21 -s llvm-openmp -- bash -c "
    cd '$d' && PYTHONPATH='$src' python '$src/cython.py' ${CYFLAGS:--3} -I '$src/tests/run' '$src/tests/run/$t.pyx' -o $t.c 2>&1 | grep -E 'rror' | head -5
    clang -O2 -w -fopenmp -fPIC -shared -I'$src/tests/run' -I\$CONDA_PREFIX/include -L\$CONDA_PREFIX/lib -Wl,-rpath,\$CONDA_PREFIX/lib -I\$(python -c 'import sysconfig;print(sysconfig.get_paths()[\"include\"])') $t.c -o $t\$(python -c 'import sysconfig;print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') || exit 3
    timeout 90 python -I -X faulthandler -c \"
import sys, doctest; sys.path.insert(0, '$d')
import $t as m
print('$t', doctest.testmod(m, verbose=False))\" 2>&1 | tail -25; echo \"$t exit=\${PIPESTATUS[0]}\""
done
