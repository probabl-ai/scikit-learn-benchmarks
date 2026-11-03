# Prior work on active wait (cakedev0 on GitHub)

Summary of what was already written in scikit-learn and the benchmark repo
before these experiments (collected 2026-11-03). Links go to the threads or
comments.

## Threads

- [scikit-learn#34764](https://github.com/scikit-learn/scikit-learn/issues/34764)
  "Impact of active wait policy of OMP implementations": knobs and defaults;
  [items 1-6](https://github.com/scikit-learn/scikit-learn/issues/34764#issuecomment-5343004000).
- [scikit-learn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935):
  the PR studied here (earlier versions #34788, #34912).
- [scikit-learn#34437](https://github.com/scikit-learn/scikit-learn/issues/34437):
  the Linux wheel bundles GCC 8.5's libgomp; recent libgomp uses
  `GOMP_SPINCOUNT=1` on hybrid CPUs.
- [scikit-learn#17334](https://github.com/scikit-learn/scikit-learn/issues/17334):
  the down side of spinning (KMeans + OpenBLAS), libgomp `sched_yield`
  prototype.
- Benchmark repo: #90 (a larger team from `predict` slows the next `fit`
  under active wait), #36 (`OMP_PROC_BIND` on hybrid CPUs), #141 (GBDT Pareto
  fronts, active vs passive).

## Facts already established

- libgomp: `GOMP_SPINCOUNT` (default 300000, read once at init); libomp:
  `KMP_BLOCKTIME` (default 200 ms, `kmp_set_blocktime` at run time). Recent
  libgomp (GCC >= 14) and libomp disable spinning on hybrid CPUs, so
  conda-forge on laptops is passive, while the PyPI wheel's old libgomp spins.
- A fit has ~10K small parallel regions. Passive wait makes HGB fits 2-10x
  slower on laptops; on the Xeon, a spin count of 3000 makes fits 5-30x slower.
- #34935 detects the wait policy from `OMP_DISPLAY_ENV=VERBOSE` in a
  subprocess (`GOMP_SPINCOUNT=[01]` or `KMP_BLOCKTIME=0`), overridable with
  `SKLEARN_OMP_WAIT_POLICY`.
- From #34788: "I initially tried dynamic sizing of the thread pool but it is
  catastrophic: OMP apparently recreates a thread pool each time it's
  resized." Hence one thread count per fit. LOG Exp 7 confirms this for
  libgomp only: libomp keeps its threads.
- Spinning hurts when threads block inside a region (GIL, locks, OpenBLAS
  mutex): a libgomp patch with `sched_yield()` every 100 spins fixed KMeans and
  kept HGB fast; "the fix belongs upstream" (not reported upstream yet).

## Ideas listed but not done

- #34764 item 1: ship libomp in the Linux wheel and call `kmp_set_blocktime`
  (risky next to pytorch, xgboost, lightgbm). Item 2: warn when active wait
  is off. Item 3: fewer parallel regions, judged then "structurally (and
  inherently) a bad candidate" for HGB: these experiments revisit it.
- Hybrid CPU handling, row-wise or samples x features chunking of the
  histograms, single-thread speed vs LightGBM, a threadpoolctl API for the
  wait policy (threadpoolctl #270/#271).
- No earlier mention of CPU idle states (C-states) or frequency: the
  mechanism in FINDINGS.md is new.
