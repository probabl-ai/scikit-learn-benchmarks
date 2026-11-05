"""The heuristic tests with the rule broken: they must fail."""
import os, pytest
import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as m

T = os.path.expanduser("~/work/hgb-regions/tcheck/test_gradient_boosting.py")
args = ["-q", "--no-header", "--tb=no", "-p", "no:cacheprovider", T, "-k", "heuristic or static_chunk"]
print("clean:", pytest.main(args))
A, P = m._MAX_N_THREADS_ACTIVE_WAIT, m._MAX_N_THREADS_PASSIVE_WAIT
m._MAX_N_THREADS_ACTIVE_WAIT, m._MAX_N_THREADS_PASSIVE_WAIT = P, A
print("tables swapped:", pytest.main(args))
m._MAX_N_THREADS_ACTIVE_WAIT, m._MAX_N_THREADS_PASSIVE_WAIT = A, P
m._MAX_N_THREADS_PASSIVE_WAIT = ((2_000_000, 1), (20_000_000, 8), (75_000_000, 4), (float("inf"), 16))
print("non monotonic stair:", pytest.main(args))
