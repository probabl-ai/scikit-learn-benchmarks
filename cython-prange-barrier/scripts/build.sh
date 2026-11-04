#!/bin/bash
# Run under benchlock, on the host. Build scikit-learn revisions against Cython trees, one private venv per pair.
# usage: build.sh JOBS NAME=SKL_REV=CYTHON_TREE...   e.g. m331=6e9d6678e7=331
W=~/work/cython-prange; cd $W; E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev; jobs=$1; shift
[ -d skl ] || git clone -q --filter=blob:none https://github.com/cakedev0/scikit-learn skl
git -C skl remote get-url upstream >/dev/null 2>&1 || git -C skl remote add upstream https://github.com/scikit-learn/scikit-learn
git -C skl fetch -q upstream main; git -C skl fetch -q origin hgb/fewer-regions hgb/fewer-regions-nb
for spec in "$@"; do IFS== read -r name rev cy <<< "$spec"
  [ -d skl-$name ] || git -C skl worktree add -q --detach ../skl-$name $rev
  if [ ! -d venv-$name ]; then
    $E/bin/python -m venv --system-site-packages venv-$name
    cp sitecustomize.py $(venv-$name/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/
  fi
  printf '#!/bin/sh\nPYTHONPATH=%s exec %s %s "$@"\n' "$W/cy/$cy" "$E/bin/python" "$W/cy/$cy/cython.py" > venv-$name/bin/cython; chmod +x venv-$name/bin/cython
  (cd skl-$name && PATH=$W/venv-$name/bin:$E/bin:$PATH pip install -q --no-build-isolation --no-deps -Cbuild-dir=build -Ccompile-args=-j$jobs . 2>&1 | grep -v notice | tail -15)
  c=$(grep -rh --include='*.c' --include='*.cpp' 'pragma omp for' skl-$name/build | sed 's/^ *//')
  b=$(find venv-$name/lib -path '*sklearn*' -name '*.so' -print0 | xargs -0 -n1 objdump -dr 2>/dev/null | grep -c 'R_X86_64.*GOMP_barrier')
  echo "$name skl=$(git -C skl-$name rev-parse --short HEAD) $(venv-$name/bin/cython -V 2>&1) GOMP_barrier-sites=$b omp-for=$(echo "$c" | wc -l) nowait=$(echo "$c" | grep -c nowait) firstprivate=$(echo "$c" | grep -c firstprivate) file=$(cd /tmp && $W/venv-$name/bin/python -c 'import sklearn; print(sklearn.__file__)')"
done
