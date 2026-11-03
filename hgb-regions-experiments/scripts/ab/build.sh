#!/bin/bash
# Run under benchlock. Builds both variants into their venvs.
set -e
W=~/work/hgb-regions
E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev
for v in exp main; do
  src=$W/skl; [ $v = main ] && src=$W/skl-main
  (cd $src && PATH=$W/venv-$v/bin:$E/bin:$PATH pip install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | grep -v "notice" | tail -20)
  $W/venv-$v/bin/python -c "import sklearn, sklearn.ensemble._hist_gradient_boosting.splitting as s; print('$v', sklearn.__file__, hasattr(s.Splitter, 'find_node_split_pair'))"
done
