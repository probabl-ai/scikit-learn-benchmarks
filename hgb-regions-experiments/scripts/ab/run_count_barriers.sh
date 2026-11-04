#!/bin/bash
# Executed barriers per parallel region of HGB. Run under benchlock. usage: run_count_barriers.sh VENV...
cd ~/work/hgb-regions
E=~/scikit-learn-benchmarks/.pixi/envs/sklearn-dev
gcc -O1 -shared -fPIC count_barriers.c -o count_barriers.so -ldl -L$E/lib -lgomp -Wl,-rpath,$E/lib || exit 1
for v in "$@"; do
  echo "=== $v"
  S=$(find $PWD/$v -name "splitting*.so")
  (cd /tmp && OMP_WAIT_POLICY=active OMP_NUM_THREADS=4 LD_PRELOAD=$HOME/work/hgb-regions/count_barriers.so:$E/lib/libgomp.so $HOME/work/hgb-regions/$v/bin/python -c "
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as H
X = np.random.RandomState(0).normal(size=(200_000, 10))
y = X[:, 0] + np.sin(X[:, 1]) + X[:, 2] * X[:, 3]
H(max_iter=10, early_stopping=False).fit(X, y)
" 2>&1 | grep -E "^COUNT .*splitting|Traceback|Error" | sort -k2,2 -k4,4 -u) > count-$v.txt
  objdump -d --no-show-raw-insn $S | awk '/^[0-9a-f]+ <.*>:/{f=$2} /call.*GOMP_barrier/{sub(":","",$1); print f, $1}' > sites-$v.txt
  objdump -t $S | awk '/_omp_fn/{print $1, $NF}' > fns-$v.txt
  python3 - "$v" <<'P'
import sys
v = sys.argv[1]
barrier_fn = {}
for l in open(f"sites-{v}.txt"):
    fn, addr = l.split()
    barrier_fn[int(addr, 16) + 5] = fn.strip("<>:").split("hist_gradient_boosting_")[-1]
fns = {int(a, 16): n.split("hist_gradient_boosting_")[-1] for a, n in (l.split() for l in open(f"fns-{v}.txt"))}
regions, barriers = {}, {}
for l in open(f"count-{v}.txt"):
    _, kind, _, off, calls = l.split()
    off, calls = int(off, 16), int(calls)
    if kind == "p":
        regions[fns.get(off, hex(off))] = calls
    else:
        fn = barrier_fn.get(off, hex(off))
        barriers.setdefault(fn, []).append(calls)
for fn, n in sorted(regions.items()):
    b = barriers.get(fn, [])
    print(f"{fn}: {n} regions, {sum(b) / n:.2f} barriers per region, sites {sorted(b, reverse=True)}")
P
done
