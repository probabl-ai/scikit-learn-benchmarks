#!/bin/bash
# Run under benchlock. Backtraces of the clang GIL hang, and a matrix of Python builds. usage: gil_hang_trace.sh OUTDIR CYTHON_SRC
HERE=$(cd "$(dirname "$0")/.." && pwd); out=$1; src=$2; mkdir -p "$out"; cd "$out"; cp "$HERE/probes/gil_hang_issue.pyx" repro.pyx
for py in python=3.13 python-freethreading=3.13; do for tc in clang gcc; do
  d=$out/${py%%=*}-$tc; mkdir -p "$d"; cp repro.pyx "$d/"
  if [ $tc = gcc ]; then spec="gcc_linux-64=15"; cc=x86_64-conda-linux-gnu-gcc; else spec="clang=21"; cc=clang; fi
  pixi exec -c conda-forge -s $py -s setuptools -s llvm-openmp -s gdb -s $spec -- bash -c '
    cd "$1"; PYTHONPATH="$2" python "$2/cython.py" repro.pyx -o repro.c >/dev/null 2>&1
    grep -h "pragma omp for" repro.c | sed "s/^ *//"
    $3 -O1 -g -w -fopenmp -fPIC -shared -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -I$(python -c "import sysconfig;print(sysconfig.get_paths()[\"include\"])") repro.c -o repro$(python -c "import sysconfig;print(sysconfig.get_config_var(\"EXT_SUFFIX\"))")
    gil=$(python -c "import sys;print(getattr(sys,\"_is_gil_enabled\",lambda:True)())")
    timeout 20 python -I -c "import sys;sys.path.insert(0,\".\");import repro;print(repro.run())" > run.txt 2>&1; rc=$?
    echo "'"${py%%=*} $tc"' gil_enabled=$gil rc=$rc output=$(tail -1 run.txt)"
    if [ $rc = 124 ]; then
      cat > drv.py <<PY
import signal, sys
sys.path.insert(0, ".")
import repro
signal.alarm(5)
print(repro.run())
PY
      timeout 120 gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex "handle SIGALRM stop nopass" -ex run -ex "thread apply all bt 14" --args python -I drv.py > bt.txt 2>&1; echo "backtrace in $1/bt.txt"
    fi' _ "$d" "$src" "$cc"
done; done
