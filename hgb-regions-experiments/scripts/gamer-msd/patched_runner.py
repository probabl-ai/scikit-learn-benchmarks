"""Run sklbench's estimator runner with HGB's fit thread count forced to $HGB_N_THREADS."""
import os, runpy, sys

n = os.environ.get("HGB_N_THREADS")
if n:
    n = int(n)
    import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as gb
    if hasattr(gb.BaseHistGradientBoosting, "_get_heuristic_optimal_n_threads"):  # PR branch
        gb.BaseHistGradientBoosting._get_heuristic_optimal_n_threads = staticmethod(lambda *a, **k: n)
    else:  # main
        gb._openmp_effective_n_threads = lambda n_threads=None: n
if os.environ.get("SHOW_POOLS"):
    import sklearn.ensemble, numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    HistGradientBoostingRegressor(max_iter=2).fit(np.random.rand(1000, 5), np.random.rand(1000))
    from threadpoolctl import threadpool_info
    for p in threadpool_info():
        print("POOL", p["user_api"], p["internal_api"], p["filepath"], p["num_threads"], file=sys.stderr)
sys.argv = ["sklbench.runners.estimator"] + sys.argv[1:]
runpy.run_module("sklbench.runners.estimator", run_name="__main__", alter_sys=True)
