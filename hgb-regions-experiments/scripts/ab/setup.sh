#!/bin/bash
# Clone the experiment branch and the PR's base commit, create one venv per build.
set -e
W=~/work/hgb-regions; cd $W
E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev
BASE=d1278c2706491eac6e67a21e4f52cbe9b3ea548e
[ -d skl ] || git clone -q --filter=blob:none https://github.com/cakedev0/scikit-learn skl
cd skl
git fetch -q origin hgb/fewer-regions && git checkout -q -B exp FETCH_HEAD
git fetch -q https://github.com/scikit-learn/scikit-learn $BASE
[ -d ../skl-main ] || git worktree add -q --detach ../skl-main $BASE
cd $W
for v in exp main; do
  if [ ! -d venv-$v ]; then
    $E/bin/python -m venv --system-site-packages venv-$v
    cp sitecustomize.py $(venv-$v/bin/python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")/
  fi
done
echo "setup ok: exp=$(git -C skl rev-parse --short HEAD) main=$(git -C skl-main rev-parse --short HEAD)"
