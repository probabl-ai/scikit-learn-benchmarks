#!/bin/bash
# Under benchlock. Build c325knob: #34935 plus knobs.patch, with Cython 3.2.5. Then check that the
# knobs do nothing when unset. On a port machine, run it inside the pixi env. usage: build_knob.sh
set -e -o pipefail
W=~/work/hgb-regions; cd $W
PR=313be4c8c8
if [ -d benchrepo ]; then
  export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
  if [ "$(uname)" = Darwin ]; then
    SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    export SDKROOT=$SDK CC=clang CXX=clang++ CFLAGS="-fopenmp -isysroot $SDK" CXXFLAGS="-fopenmp -isysroot $SDK"
  fi
  PY=python; BINPATH=$W/venv-c325knob/bin:$PATH
else
  E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev
  PY=$E/bin/python; BINPATH=$W/venv-c325knob/bin:$E/bin:$PATH
fi
[ -d skl-c325knob ] || git -C skl worktree add -q --detach ../skl-c325knob $PR
git -C skl-c325knob checkout -q -f --detach $PR
git -C skl-c325knob apply $W/knobs.patch
if [ ! -d venv-c325knob ]; then
  $PY -m venv --system-site-packages venv-c325knob
  if [ ! -d benchrepo ]; then
    cp sitecustomize.py $(venv-c325knob/bin/python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")/
    venv-c325knob/bin/pip install -q "cython==3.2.5" 2>&1 | { grep -v notice || true; }
  fi
fi
(cd skl-c325knob && PATH=$BINPATH pip install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | { grep -v notice || true; } | tail -20)
C=$(find skl-c325knob/build -name "splitting.c" | head -1)
echo "c325knob | $(head -1 $C | cut -c1-40)"
cd /tmp
OMP_WAIT_POLICY=passive $W/venv-c325knob/bin/python $W/identity.py $W/identity-c325knob.json
HGB_T=2 HGB_SM=1 HGB_FS=0 OMP_WAIT_POLICY=passive $W/venv-c325knob/bin/python $W/identity.py $W/identity-c325knob-on.json
cd $W
venv-c325knob/bin/python - <<PY
import json
a = {v: json.load(open(f"identity-{v}.json")) for v in ("c325pr", "c325knob", "c325knob-on")}
print("knobs unset == #34935:", sum(a["c325knob"][k] == a["c325pr"][k] for k in a["c325pr"]), "of", len(a["c325pr"]))
print("knobs set == #34935:", sum(a["c325knob-on"][k] == a["c325pr"][k] for k in a["c325pr"]), "of", len(a["c325pr"]))
PY
cd /tmp
HGB_T=2 HGB_SM=1 HGB_FS=0 OMP_WAIT_POLICY=passive $W/venv-c325knob/bin/python - <<PY
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as R
from sklearn.ensemble._hist_gradient_boosting import grower, splitting
seen = {}
init = grower.TreeGrower.__init__
def wrapped(self, *a, **k):
    init(self, *a, **k)
    s = self.splitter
    seen.update(n_threads=s.n_threads, sm=s.knob_split_min, fs=s.knob_fs_factor, active=s.active_wait)
grower.TreeGrower.__init__ = wrapped
R(max_iter=1).fit(np.random.rand(20000, 5), np.random.rand(20000))
print("seen with knobs set", seen)
PY
echo "### build done"
