#!/bin/bash
# Under benchlock. Build c325retune: #34935 plus retune.patch, with Cython 3.2.5. Then check trees
# and predictions against #34935. On a port machine, run it inside the pixi env. usage: build_retune.sh
set -e -o pipefail
W=~/work/hgb-regions; cd $W
PR=313be4c8c8
if [ -d benchrepo ]; then
  export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
  if [ "$(uname)" = Darwin ]; then
    SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    export SDKROOT=$SDK CC=clang CXX=clang++ CFLAGS="-fopenmp -isysroot $SDK" CXXFLAGS="-fopenmp -isysroot $SDK"
  fi
  PY=python; BINPATH=$W/venv-c325retune/bin:$PATH
else
  E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev
  PY=$E/bin/python; BINPATH=$W/venv-c325retune/bin:$E/bin:$PATH
fi
[ -d skl-c325retune ] || git -C skl worktree add -q --detach ../skl-c325retune $PR
git -C skl-c325retune checkout -q -f --detach $PR
git -C skl-c325retune apply $W/retune.patch
if [ ! -d venv-c325retune ]; then
  $PY -m venv --system-site-packages venv-c325retune
  if [ ! -d benchrepo ]; then
    cp sitecustomize.py $(venv-c325retune/bin/python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")/
    venv-c325retune/bin/pip install -q "cython==3.2.5" 2>&1 | { grep -v notice || true; }
  fi
fi
(cd skl-c325retune && PATH=$BINPATH pip install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | { grep -v notice || true; } | tail -20)
cd /tmp
OMP_WAIT_POLICY=passive $W/venv-c325retune/bin/python $W/identity.py $W/identity-c325retune.json
cd $W
venv-c325retune/bin/python - <<PY
import json
a, b = (json.load(open(f"identity-{v}.json")) for v in ("c325retune", "c325pr"))
print("retune == #34935:", sum(a[k] == b[k] for k in b), "of", len(b))
PY
echo "### build done"
