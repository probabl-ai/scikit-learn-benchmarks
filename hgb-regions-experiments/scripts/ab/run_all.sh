#!/bin/bash
# Run under benchlock: update + rebuild both builds, then bench.py once per profile.
# usage: run_all.sh THREADS "PROFILE|VARIANTS|KEY=VAL KEY=VAL" ...   (VARIANTS empty = all)
W=~/work/hgb-regions; cd $W
bash setup.sh && bash build.sh || exit 1
export PYTHONPATH=~/scikit-learn-benchmarks
T=$1; shift
for spec in "$@"; do
  IFS='|' read -r prof variants envs <<< "$spec"
  eargs=(); for kv in $envs; do eargs+=(--env "$kv"); done
  vargs=(); [ -n "$variants" ] && vargs=(--variants "$variants")
  echo "### $prof $(date -Is)"
  env $envs OMP_NUM_THREADS=$T venv-exp/bin/python -c "
from sklearn.utils._openmp_helpers import _openmp_uses_active_wait; import check_rt
print('active_wait detected:', _openmp_uses_active_wait())"
  venv-main/bin/python bench.py --profile $prof --threads $T "${vargs[@]}" "${eargs[@]}" --out results_$prof.jsonl
done
echo "### all done $(date -Is)"
