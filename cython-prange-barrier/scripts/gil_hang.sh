#!/bin/bash
# Run under benchlock. prange holding the GIL (GIL build of Python), with and without a read of the target after the loop.
# usage: gil_hang.sh OUTDIR LABEL=CYTHON_SRC...
HERE=$(cd "$(dirname "$0")/.." && pwd); out=$1; shift
for pair in "$@"; do label=${pair%%=*}; src=${pair#*=}
for tc in gcc clang; do
  d=$out/$label-$tc; mkdir -p "$d"; cp "$HERE/probes/gil_hang.pyx" "$d/"
  if [ $tc = gcc ]; then spec="gcc_linux-64=15"; cc=x86_64-conda-linux-gnu-gcc; else spec="clang=21"; cc=clang; fi
  pixi exec -c conda-forge -s python=3.13 -s setuptools -s llvm-openmp -s $spec -- bash -c '
    cd "$1" && PYTHONPATH="$2" python "$2/cython.py" gil_hang.pyx -o gil_hang.c >/dev/null 2>&1
    grep -h "#pragma omp for" gil_hang.c | sed "s/^ *//"
    $3 -O2 -w -fopenmp -fPIC -shared -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -I$(python -c "import sysconfig;print(sysconfig.get_paths()[\"include\"])") gil_hang.c -o gil_hang$(python -c "import sysconfig;print(sysconfig.get_config_var(\"EXT_SUFFIX\"))") && python -I "$4" "$1" "$5"' _ "$d" "$src" "$cc" "$HERE/scripts/gil_hang_run.py" "$label-$tc"
done; done
