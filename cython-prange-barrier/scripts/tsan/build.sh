#!/bin/bash
# In the pixi env, under benchlock. Build scikit-learn with TSan against a Cython tree. usage: build.sh LABEL SKL_REV CYTHON_TREE
W=$(cd "$(dirname "$0")" && pwd); cd "$W"
label=$1; rev=$2; tree=$W/../cy/$3
[ -d skl ] || git clone -q --filter=blob:none https://github.com/cakedev0/scikit-learn skl
git -C skl remote get-url upstream >/dev/null 2>&1 || git -C skl remote add upstream https://github.com/scikit-learn/scikit-learn
git -C skl fetch -q upstream main
git -C skl fetch -q origin hgb/fewer-regions
[ -d skl-$label ] || git -C skl worktree add -q --detach ../skl-$label $rev
[ -d venv-$label ] || python -m venv --system-site-packages venv-$label
printf '#!/bin/sh\nPYTHONPATH=%s exec %s %s "$@"\n' "$tree" "$(command -v python)" "$tree/cython.py" > venv-$label/bin/cython
chmod +x venv-$label/bin/cython
export CC=clang CXX=clang++ CFLAGS="-fopenmp -fsanitize=thread -g -O1" CXXFLAGS="-fopenmp -fsanitize=thread -g -O1"
mkdir -p logs
rm -rf skl-$label/build
# PYTHONPATH: scikit-learn's Tempita step imports Cython.
(cd skl-$label && PYTHONPATH=$tree PATH=$W/venv-$label/bin:$PATH "$W/venv-$label/bin/pip" install -q --no-build-isolation --no-deps -Cbuild-dir=build . > "$W/logs/build-$label.log" 2>&1)
grep -nE "ERROR|error:|Error" logs/build-$label.log | cut -c1-200 | head -15
pragmas=$(grep -rh --include='*.c' --include='*.cpp' 'pragma omp for' skl-$label/build | sed 's/^ *//')
echo "$label skl=$(git -C skl-$label rev-parse --short HEAD) $(venv-$label/bin/cython -V 2>&1) omp-for=$(echo "$pragmas" | wc -l) nowait=$(echo "$pragmas" | grep -c nowait) firstprivate=$(echo "$pragmas" | grep -c firstprivate)"
