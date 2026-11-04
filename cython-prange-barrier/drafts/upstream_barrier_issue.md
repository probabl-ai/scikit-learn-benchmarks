# [ENH] Avoid the barrier at the start of prange when firstprivate + lastprivate can't be observed

Draft, not posted anywhere. Should come after the end barrier bug is sorted out, since the patch
builds on it.

----

Every `prange` starts with a barrier on GCC and clang, even when nothing depends on it. All the threads wait for the slowest one to wake up before any of them starts working, which is expensive with a passive OpenMP runtime. That's the default on Intel hybrid CPUs: libgomp sets a spin count of 1 there and libomp a blocktime of 0.

The barrier comes from the clauses Cython generates for the target and for every other non-Python private:

```c
#pragma omp for nowait firstprivate(i) lastprivate(i) schedule(static)
```

OpenMP requires the `lastprivate` update to happen after all the `firstprivate` initializations (5.2, section 5.4.5), and GCC and clang implement that with a barrier. Neither clause alone does this on GCC, and with `private(i)` there's no barrier on either compiler. On the target, `firstprivate` was added in 2011 to silence a warning, which GCC itself has suppressed since 2016 (PR 72781). On other variables, it's what lets them be initialized before the loop since GH-7178, so it can't just be dropped.

I propose to generate `private(x)` when flow control shows that nothing can observe the difference: every read in the loop body only sees assignments of the same body, and no read outside of the body sees one of them. Variables in closures, variables whose address is taken and non-C types keep both clauses. In scikit-learn, that's 398 of the 399 `omp for`, with no change to its code.

I also propose to keep `nowait` on a `prange` that ends a `parallel` block, since only the end of the region follows it. GCC doesn't always remove that barrier by itself and clang doesn't try.

Fit time of `HistGradientBoostingRegressor` on `year_prediction_msd`, in s, median of 3:

| machine | threads | wait policy | 3.3.1 | patched |
| --- | --- | --- | --- | --- |
| Xeon, one socket | 86 | passive | 53.09 | 33.72 |
| Xeon, one socket | 86 | default | 4.20 | 3.57 |
| Core Ultra 9 285 | 24 | default (passive), libgomp | 20.48 | 15.95 |
| Core Ultra 9 285 | 24 | default (passive), libomp | 17.33 | 12.06 |
| Core Ultra X7 358H | 16 | default (passive), libgomp | 12.18 | 10.25 |

With its parallel regions merged, HGB goes from 7.62 s to 5.97 s on the Core Ultra 9 with libgomp, the same as replacing `prange` by `parallel` blocks with manual chunks (5.98 s).

It's not a uniform gain: `kddcup09_churn` gets 4-16% slower on the Core Ultra 9, and I suspect idle states on mixed P and E cores. Results don't change, and scikit-learn's tests of the modules using `prange` give the same counts.

The patch adds a flow control check in `ParallelRangeNode` and a test file that checks the generated clauses for each case. Cython's parallel tests pass in C and C++ with GCC 15. They also pass with clang 21, built by hand since `runtests.py` doesn't enable OpenMP on clang, except `test_pure_mode`, which needs `Cython` to be importable and fails the same way on master in that setup.

Would you be interested in a PR? The part I'm least sure about is whether relying on `cf_state` this way is acceptable to you.
