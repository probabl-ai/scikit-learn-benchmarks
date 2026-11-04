#!/bin/bash
# Run under benchlock, on the host: scikit-learn's tests of the modules using prange, on the fixed builds.
cd /tmp
W=~/work/cython-prange
mods="sklearn.ensemble._hist_gradient_boosting sklearn.cluster.tests.test_k_means sklearn.metrics.tests.test_pairwise_distances_reduction sklearn.metrics.tests.test_pairwise sklearn._loss sklearn.manifold.tests.test_t_sne"
echo "== mB2, main"
$W/venv-mB2/bin/python -m pytest -q -p no:cacheprovider --pyargs $mods 2>&1 | tail -6
echo "== fB2, fuse3"
SKLEARN_HGB_EXP=fuse3 $W/venv-fB2/bin/python -m pytest -q -p no:cacheprovider --pyargs sklearn.ensemble._hist_gradient_boosting 2>&1 | tail -6
echo "== m331, baseline"
$W/venv-m331/bin/python -m pytest -q -p no:cacheprovider --pyargs $mods 2>&1 | tail -6
