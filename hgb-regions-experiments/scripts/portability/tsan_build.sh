#!/bin/bash
# TSan build of hgb/merge-regions, in a separate venv. Run under benchlock, in the pixi env.
W=$(cd "$(dirname "$0")" && pwd); cd "$W"
L=${1:-tsan}; BRANCH=${2:-hgb/merge-regions}
[ -d skl-$L ] || git clone -q --filter=blob:none --branch $BRANCH https://github.com/cakedev0/scikit-learn skl-$L
[ -d venv-$L ] || python -m venv --system-site-packages venv-$L
export CC=clang CFLAGS="-fopenmp -fsanitize=thread -g -O1"  # C only: C++ extensions stay uninstrumented
(cd skl-$L && "$W/venv-$L/bin/pip" install -q --no-build-isolation --no-deps -Cbuild-dir=build . 2>&1 | grep -vE "notice|^$" | tail -15)
echo built
