#!/bin/bash
# Run under benchlock, in the pixi env. Build main (base), #34935 and hgb/merge-regions with
# Cython 3.2.5, on a machine without the benchmarks repo. usage: pixi run bash build_port.sh
set -e -o pipefail
W=~/work/hgb-regions; cd $W
BASE=d1278c2706491eac6e67a21e4f52cbe9b3ea548e
# Otherwise meson's Cython check can find the system Python's headers.
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
if [ "$(uname)" = Darwin ]; then
  # The Command Line Tools' SDK doesn't work with this Xcode's linker.
  SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
  export SDKROOT=$SDK CC=clang CXX=clang++ CFLAGS="-fopenmp -isysroot $SDK" CXXFLAGS="-fopenmp -isysroot $SDK"
fi
[ -d skl ] || git clone -q --filter=blob:none --no-checkout https://github.com/cakedev0/scikit-learn skl
git -C skl fetch -q https://github.com/scikit-learn/scikit-learn $BASE
git -C skl fetch -q origin hgb/active_wait; PR=$(git -C skl rev-parse FETCH_HEAD)
git -C skl fetch -q origin hgb/merge-regions; CLEAN=$(git -C skl rev-parse FETCH_HEAD)
for spec in "c325main $BASE" "c325pr $PR" "c325clean $CLEAN"; do
  read -r name sha <<< "$spec"
  [ -d skl-$name ] || git -C skl worktree add -q --detach ../skl-$name $sha
  git -C skl-$name checkout -q --detach $sha
  [ -d venv-$name ] || python -m venv --system-site-packages venv-$name
  (cd skl-$name && PATH=$W/venv-$name/bin:$PATH pip install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | { grep -v notice || true; } | tail -20)
  C=$(find skl-$name/build -name "splitting.c" | head -1)
  echo "$name $(git -C skl-$name log --oneline -1 | cut -c1-60) | $(head -1 $C | cut -c1-40)"
done
cd /tmp
for v in c325main c325pr c325clean; do
  OMP_WAIT_POLICY=passive $W/venv-$v/bin/python $W/identity.py $W/identity-$v.json
done
cd $W
venv-c325main/bin/python - <<PY
import json
a = {v: json.load(open(f"identity-{v}.json")) for v in ("c325main", "c325pr", "c325clean")}
n = sum(a["c325clean"][k] == a["c325pr"][k] == a["c325main"][k] for k in a["c325main"])
print("identical", n, "of", len(a["c325main"]))
PY
cd /tmp
OMP_NUM_THREADS=2 $W/venv-c325clean/bin/python -m pytest --pyargs sklearn.ensemble._hist_gradient_boosting -q -p no:cacheprovider 2>&1 | tail -1
$W/venv-c325pr/bin/python - <<PY
import numpy as np, joblib
from threadpoolctl import threadpool_info
from sklearn.ensemble import HistGradientBoostingRegressor as R
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import _openmp_uses_active_wait
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
R(max_iter=1).fit(np.random.rand(100, 2), np.random.rand(100))
print("openmp", [(p["filepath"].split("/")[-1], p["version"]) for p in threadpool_info() if p["user_api"] == "openmp"],
      "effective threads", _openmp_effective_n_threads(), "physical", joblib.cpu_count(only_physical_cores=True),
      "active wait", _openmp_uses_active_wait())
PY
OMP_WAIT_POLICY=passive $W/venv-c325pr/bin/python -c "
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import _openmp_uses_active_wait as f; print('with OMP_WAIT_POLICY=passive: active wait', f())"
echo "### build done"
