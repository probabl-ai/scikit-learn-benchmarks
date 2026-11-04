#!/bin/bash
# Run under benchlock. usage: build_branch.sh NAME BRANCH: build a cakedev0 branch into venv-NAME.
set -e
W=~/work/hgb-regions; cd $W
E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev
name=$1; branch=$2
git -C skl fetch -q origin "$branch"
sha=$(git -C skl rev-parse FETCH_HEAD)
[ -d skl-$name ] || git -C skl worktree add -q --detach ../skl-$name $sha
git -C skl-$name checkout -q --detach $sha
if [ ! -d venv-$name ]; then
  $E/bin/python -m venv --system-site-packages venv-$name
  cp sitecustomize.py $(venv-$name/bin/python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")/
fi
(cd skl-$name && PATH=$W/venv-$name/bin:$E/bin:$PATH pip install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | grep -v notice | tail -40)
venv-$name/bin/python -c "import sklearn; print('$name', sklearn.__file__)"
echo "$name built at $(git -C skl-$name rev-parse --short HEAD)"
