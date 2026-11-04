#!/bin/bash
# Run under benchlock. Build main (base), the PR and hgb/merge-regions with Cython 3.2.5, the
# latest version allowed by scikit-learn's pin. usage: build_cython325.sh
set -e
W=~/work/hgb-regions; cd $W
E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev
BASE=d1278c2706491eac6e67a21e4f52cbe9b3ea548e
git -C skl fetch -q origin hgb/active_wait; PR=$(git -C skl rev-parse FETCH_HEAD)
git -C skl fetch -q origin hgb/merge-regions; CLEAN=$(git -C skl rev-parse FETCH_HEAD)
for spec in "c325main $BASE" "c325pr $PR" "c325clean $CLEAN"; do
  read -r name sha <<< "$spec"
  [ -d skl-$name ] || git -C skl worktree add -q --detach ../skl-$name $sha
  git -C skl-$name checkout -q --detach $sha
  if [ ! -d venv-$name ]; then
    $E/bin/python -m venv --system-site-packages venv-$name
    cp sitecustomize.py $(venv-$name/bin/python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")/
    venv-$name/bin/pip install -q "cython==3.2.5" 2>&1 | grep -v notice || true
  fi
  (cd skl-$name && PATH=$W/venv-$name/bin:$E/bin:$PATH pip install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | grep -v notice | tail -20)
  C=$(find skl-$name/build -name "splitting.c" | head -1)
  echo "$name $(git -C skl-$name log --oneline -1 | cut -c1-60) | $(head -1 $C | cut -c1-40)"
  S=$(dirname $(find venv-$name -name "splitting*.so" | head -1))
  objdump -d $S/splitting*.so | awk '/^[0-9a-f]+ <.*>:/{f=$2} /call.*GOMP_barrier/{print f}' | sed 's/__pyx_pf_7sklearn_8ensemble_23_hist_gradient_boosting_//' | sort | uniq -c | cut -c1-80
done
for v in c325main c325pr c325clean; do
  (cd /tmp && OMP_WAIT_POLICY=passive $W/venv-$v/bin/python $W/identity.py $W/identity-$v.json)
done
venv-main/bin/python - <<PY
import json
a = {v: json.load(open(f"identity-{v}.json")) for v in ("c325main", "c325pr", "c325clean")}
n = sum(a["c325clean"][k] == a["c325pr"][k] == a["c325main"][k] for k in a["c325main"])
print("identical", n, "of", len(a["c325main"]))
PY
