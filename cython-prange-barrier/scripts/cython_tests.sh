#!/bin/bash
# Run under benchlock. The parallel tests of a Cython tree: C and C++ with GCC 15, then clang 21 by hand.
# usage: cython_tests.sh CYTHON_SRC OUTDIR
HERE=$(cd "$(dirname "$0")" && pwd); src=$1; out=$2; mkdir -p "$out"
tests=('run\.parallel$' 'run\.parallel_with_gil' 'run\.sequential_parallel' 'run\.parallel_swap' 'run\.parallel_private_clauses')
cd "$src" || exit 1
for lang in c cpp; do
  if [ $lang = c ]; then flag=--no-cpp; extra=('run\.pure_parallel' 'errors\.e_cython_parallel'); else flag=--no-c; extra=('run\.cpp_locals_parallel'); fi
  timeout 300 pixi exec -c conda-forge -s python=3.13 -s setuptools -s gcc=15 -s gxx=15 -- \
    python runtests.py $flag --no-code-style --no-refnanny --no-unit --no-doctest --no-pyregr --no-examples -j4 \
    "${tests[@]}" "${extra[@]}" > "$out/gcc-$lang.log" 2>&1
  echo "gcc $lang exit=$?"
  grep -E '^(Ran [1-9]|FAILED|FAIL:|ERROR:)' "$out/gcc-$lang.log" | sort | uniq -c
done
rm -rf TEST_TMP
CYFLAGS=-2 bash "$HERE/doctests_clang.sh" "$src" "$out/clang" parallel 2>&1 | grep -E 'TestResults|exit='
bash "$HERE/doctests_clang.sh" "$src" "$out/clang" parallel_with_gil parallel_private_clauses 2>&1 | grep -E 'TestResults|exit='
