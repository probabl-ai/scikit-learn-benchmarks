# Findings

Every `prange` starts with a barrier on GCC and clang. It can be removed safely in most loops,
and that makes HGB fits up to ~35% faster with a passive runtime (`year_prediction_msd`, one
Xeon socket), with no change of results. Other cases gain less, and a few get slower. Details
and commands are in `LOG.md`.

## Why the barrier is there

OpenMP requires the `lastprivate` update to come after all the `firstprivate` initializations
of a variable in both clauses (5.2, section 5.4.5). GCC and clang implement that with a barrier
at the start of the loop. On `omp for nowait`:

| compiler | `firstprivate` | `lastprivate` | both | `private` |
| --- | --- | --- | --- | --- |
| GCC 7.5 to 15.3 | 0 | 0 | 1 (start) | 0 |
| clang 11 to 21 | 0 | 1 (end) | 2 (start, end) | 0 |
| MSVC latest, compile only | 0 | 0 | 0 | 0 |

Cython puts both clauses on the target since 2011, to silence a warning that GCC itself silenced
in 2016, and on all the other variables since 3.2 (GH-7178), on purpose, so that they can be
initialized before the loop.

The passive default on hybrid CPUs is confirmed in the sources. libgomp sets a spin count of 1
when CPUID reports a hybrid CPU (`libgomp/config/linux/x86/spincount.h`). libomp's default
blocktime is 0 on them (`KMP_DEFAULT_BLOCKTIME` in `kmp.h`).

## Two bugs found on the way

Both are on the fork only for now.

- [cakedev0/cython#1](https://github.com/cakedev0/cython/issues/1): since 3.3, code after a
  `prange` in a nogil `parallel` block can run before the other threads finish, with GCC.
- [cakedev0/cython#2](https://github.com/cakedev0/cython/issues/2): since 3.3, a `prange` holding
  the GIL hangs with clang on GIL builds. `runtests.py` doesn't enable OpenMP on clang, which is
  probably why nobody saw it.

## The fix

Local Cython branches, on upstream master 7b3aab363.

- Fix A, `prange-fix-end-barrier`: `nowait` only when a barrier isn't needed. Fixes #1.
- Fix B, `prange-no-start-barrier`, on top of A: `private(x)` when flow control shows that nothing
  can observe the difference, so 398 of the 399 `omp for` of scikit-learn. And `nowait` on the
  last statement of a `parallel` block, since only the end of the region follows it.

Tests: Cython's parallel tests pass in C and C++ with GCC 15, and by hand with clang 21 except 2
`test_pure_mode` failures from my harness (same on master). scikit-learn's tests of the modules
using `prange` give the same counts as the baseline: 2984 passed, 1061 skipped, 3 xfailed. HGB
with `fuse3`: 365 passed, 105 skipped. Results of the A/B cases agree across builds (below 2e-15).

## Timings

Median fit in s, 3 fits per variant. `gamer` is 24 threads, `laptop` 16, `xeon` one socket with
86 threads. `pr` is the unmerged HGB, `fuse3` the merged one.

| year_prediction_msd | `pr` 3.3.1 | `pr` fixed | `fuse3` 3.3.1 | `fuse3` fixed | `fuse3` manual chunks |
| --- | --- | --- | --- | --- | --- |
| gamer, libgomp | 20.48 | 15.95 | 7.62 | 5.97 | 5.98 |
| gamer, libomp | 17.33 | 12.06 | 7.16 | 5.39 | 5.80 |
| laptop, libgomp | 12.18 | 10.25 | 4.68 | 4.54 | 4.48 |
| xeon, passive | 53.09 | 33.72 | 18.14 | 13.23 | 14.22 |
| xeon, default (spins) | 4.20 | 3.57 | 2.91 | 2.58 | 2.51 |

So the fix reaches the manual chunks without changing the `prange` code.

| other cases, median of 6, in s | 3.2.4 | 3.3.1 | fixed |
| --- | --- | --- | --- |
| KMeans lloyd, xeon passive | 0.667 | 0.520 | 0.600 |
| KMeans lloyd, xeon default | 0.526 | 0.459 | 0.417 |
| KMeans lloyd, gamer libgomp | 0.184 | 0.173 | 0.182 |
| KMeans elkan, xeon passive | 1.028 | 0.815 | 0.983 |
| ArgKmin, gamer libgomp | 0.116 | 0.116 | 0.097 |
| ArgKmin, xeon passive | 0.250 | 0.265 | 0.210 |
| t-SNE, gamer libgomp | 0.899 | 0.777 | 0.775 |
| HGB predict, xeon passive | 3.185 | 3.072 | 2.126 |
| HGB predict, gamer libomp | 0.077 | 0.072 | 0.047 |

scikit-learn pins `cython<3.2.6`, so 3.2.4 is the fair baseline for users. Against it I saw no
slowdown on KMeans. Against 3.3.1, KMeans with a passive runtime is 15-21% slower on the Xeon: 3.3
doesn't wait at the end of the loop, and KMeans doesn't need that wait since it then reads
thread-local buffers. But Cython can't know that.

## Where it doesn't help

- `kddcup09_churn` on gamer: 4-16% slower with the fix, in the 4 configurations. On the Xeon it
  gains. Probably idle states on mixed P and E cores, but I haven't checked.
- The laptop is noisy: its first round showed HGB predict 27% faster and KMeans lloyd 11% slower,
  and neither came back in the second round with 2x the reps.

## Not done

- Nothing was run on Windows, and no scikit-learn build was made with clang.
- I didn't audit scikit-learn's 31 loops in `parallel` blocks for reads after the loop. It only
  matters for builds using 3.3 (the pixi `sklearn-dev` env), since Fix A restores the barrier.
- The third barrier call site of `with_parallel_two` in `results/local/barriers_generated.txt`
  isn't explained.
