#!/bin/bash
# Run under benchlock. Generate C from probes/warn.pyx, strip clauses, compile with -Wall -Wextra, count warnings.
# usage: warn_matrix.sh OUTDIR CYTHON_SRC
HERE=$(cd "$(dirname "$0")/.." && pwd); out=$1; cysrc=$2; mkdir -p "$out"; cd "$out"
PY="python=3.13 setuptools numpy llvm-openmp"
pxe() { local a=(); for s in $PY $1; do a+=(-s "$s"); done; shift; pixi exec -c conda-forge "${a[@]}" -- "$@"; }
pxe "" env PYTHONPATH="$cysrc" python "$cysrc/cython.py" -3 "$HERE/probes/warn.pyx" -o warn.c || exit 1
grep -n '#pragma omp for' warn.c > pragmas.txt; cat pragmas.txt
sed -E '/#pragma omp for/ s/ firstprivate\(__pyx_v_[a-z0-9_]+\)//g' warn.c > warn_nofirst.c
sed -E '/#pragma omp for/ { s/ firstprivate\(__pyx_v_[a-z0-9_]+\)//g; s/ lastprivate\((__pyx_v_[a-z0-9_]+)\)/ private(\1)/g }' warn.c > warn_private.c
for cc in "gcc_linux-64=8 x86_64-conda-linux-gnu-gcc" "gcc_linux-64=11 x86_64-conda-linux-gnu-gcc" "gcc_linux-64=13 x86_64-conda-linux-gnu-gcc" "gcc_linux-64=15 x86_64-conda-linux-gnu-gcc" "clang=14 clang" "clang=21 clang"; do
  spec=${cc% *}; bin=${cc#* }
  for src in warn warn_nofirst warn_private; do for opt in O2 O3; do
    log="$src.${spec%% *}.$opt.log"; log=${log//=/-}
    pxe "$spec" bash -c "\$0 -$opt -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-sign-compare -Wno-unused-command-line-argument -fopenmp -fPIC -I\$CONDA_PREFIX/include -I\$(python -c 'import sysconfig;print(sysconfig.get_paths()[\"include\"])') -c $src.c -o /dev/null" "$bin" > "$log" 2>&1
    printf '%-16s %-14s %s  errors=%s pyx_v_uninit=%s other_uninit=%s  %s\n' "$spec" "$src" "$opt" "$(grep -c ' error:' "$log")" "$(grep -E 'uninitialized' "$log" | grep -c '__pyx_v_')" "$(grep -E 'uninitialized' "$log" | grep -vc '__pyx_v_')" "$(grep -E 'uninitialized' "$log" | grep -oE '__pyx_[a-z]+_[A-Za-z0-9_]+' | sort -u | head -5 | tr '\n' ' ')"
  done; done
done
