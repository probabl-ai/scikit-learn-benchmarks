"""The HGB tests on the retuned rule, then the heuristic tests with the rule broken: they must fail."""
import sys, pytest
import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as m

TESTS = "sklearn.ensemble._hist_gradient_boosting.tests.test_gradient_boosting"
print("clean, all HGB tests:", pytest.main(["-q", "--no-header", "--pyargs", "sklearn.ensemble._hist_gradient_boosting"]))
A, P = m._MAX_N_THREADS_ACTIVE_WAIT, m._MAX_N_THREADS_PASSIVE_WAIT
m._MAX_N_THREADS_ACTIVE_WAIT, m._MAX_N_THREADS_PASSIVE_WAIT = P, A
print("tables swapped:", pytest.main(["-q", "--no-header", "--tb=no", "--pyargs", TESTS, "-k", "heuristic or static_chunk"]))
m._MAX_N_THREADS_ACTIVE_WAIT, m._MAX_N_THREADS_PASSIVE_WAIT = A, P
m._MAX_N_THREADS_PASSIVE_WAIT = ((2_000_000, 1), (20_000_000, 8), (75_000_000, 4), (float("inf"), 16))
print("non monotonic stair:", pytest.main(["-q", "--no-header", "--tb=no", "--pyargs", TESTS, "-k", "heuristic or static_chunk"]))
