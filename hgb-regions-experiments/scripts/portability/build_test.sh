#!/bin/bash
# Build hgb/merge-regions into a venv and run the HGB tests. Run under benchlock, in the pixi env.
# usage: build_test.sh LABEL CC [CFLAGS]
label=$1; export CC=$2; export CFLAGS=${3:-}
W=$(cd "$(dirname "$0")" && pwd)
cd "$W"
[ -d skl-$label ] || git clone -q --filter=blob:none --branch hgb/merge-regions https://github.com/cakedev0/scikit-learn skl-$label
git -C skl-$label fetch -q origin hgb/merge-regions && git -C skl-$label checkout -q FETCH_HEAD
echo "== $label at $(git -C skl-$label log --oneline -1), CC=$CC CFLAGS=$CFLAGS"
"$CC" --version | head -1
[ -d venv-$label ] || python -m venv --system-site-packages venv-$label
(cd skl-$label && "$W/venv-$label/bin/pip" install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | grep -vE "notice|^$" | tail -15)
cd /tmp
"$W/venv-$label/bin/python" - <<PY
import sklearn
from threadpoolctl import threadpool_info
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as R
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
R(max_iter=1).fit(np.random.rand(100, 2), np.random.rand(100))
print("sklearn", sklearn.__file__)
print("openmp runtimes", [p["filepath"].split("/")[-1] for p in threadpool_info() if p["user_api"] == "openmp"], "effective threads", _openmp_effective_n_threads())
PY
OMP_NUM_THREADS=${TEST_THREADS:-4} "$W/venv-$label/bin/python" -m pytest --pyargs sklearn.ensemble._hist_gradient_boosting -q -p no:cacheprovider 2>&1 | tail -${TEST_TAIL:-12}
