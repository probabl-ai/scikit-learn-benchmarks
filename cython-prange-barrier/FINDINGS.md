# Findings

Every `prange` starts with a barrier on GCC and clang. It can be removed safely in most loops,
and that makes HGB fits up to ~35% faster with a passive runtime (`year_prediction_msd`, one
Xeon socket), with no change of results. Other cases gain less, and a few get slower: HGB on
scikit-learn main with libomp on the Core Ultra 9 is ~23% slower with Fix B. Details and commands
are in `LOG.md`.

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

Impact on scikit-learn main (d5a61d1149cb), from reading the code with
`scripts/audit_parallel_blocks.py`, not from TSan. Its pin (`cython<3.2.6`) keeps wheels away from
both bugs for now.

- #1: I found no wrong results in the 31 loops in `parallel` blocks. What follows them is `free()`
  of per-thread buffers, a reduction of per-thread buffers under a lock (KMeans), an empty hook, or
  a second loop. The second loop reads the first one's writes only in `_radius_neighbors` and
  `_radius_neighbors_classmode`, at the same index with `schedule='static'` over the same range.
- #2: not affected. None of the 17 generated files with a `prange` use the GIL-holding path.

## The fix

PRs on the fork, on upstream master 7b3aab363.

- Fix A, [cakedev0/cython#3](https://github.com/cakedev0/cython/pull/3): `nowait` only when a
  barrier isn't needed: with the GIL held, on a standalone `prange`, and on the last statement of
  a `parallel` block, since only the end of the region follows it. Fixes #1.
- Fix B, [cakedev0/cython#4](https://github.com/cakedev0/cython/pull/4), on top of A: `private(x)`
  when flow control shows that nothing can observe the difference, so 398 of the 399 `omp for` of
  scikit-learn.

Tests: Cython's parallel tests pass in C and C++ with GCC 15, and by hand with clang 21 except 2
`test_pure_mode` failures from my harness (same on master). scikit-learn's tests of the modules
using `prange` give the same counts as the baseline: 2984 passed, 1061 skipped, 3 xfailed. HGB
with `fuse3`: 365 passed, 105 skipped. Results of the A/B cases agree across builds (below 2e-15).

TSan with Archer finds no race in the fixes, with clang 23.1 and libomp 23.1, 4 threads
(`scripts/tsan/`, `results/tsan/`, method from the HGB session):

| Cython | scikit-learn workload, 17 cases | small race probe |
| --- | --- | --- |
| 3.3.1 | 0 reports | 0 reports |
| Fix A | 0 reports | 0 reports |
| Fix B | 0 reports | 0 reports |
| Fix B on `fuse3` | 0 reports | |
| unsafe control: `nowait` on every `prange` | crashes on `fuse3` | 4 reports, wrong values |

The control shows that this setup catches the bug. The 17 values are identical across builds. It's
one run on small data, so it shows these paths are clean, not that no race can exist. TSan doesn't
cover GCC builds, where 3.3 has the race. With Fix B both compilers get the same barriers from
`private`, but I didn't run TSan on GCC.

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

HGB fits on scikit-learn main (d5a61d1149cb), not one of the HGB branches above. Median of 5 fits in
s, interleaved, models identical (same R2 where recorded, same `n_iter`):

| year_prediction_msd | 3.2.4 | 3.3.1 | Fix A | Fix B |
| --- | --- | --- | --- | --- |
| gamer, libgomp | 14.05 | 12.88 | 13.18 | 11.86 |
| gamer, libomp | 6.66 | 5.55 | 5.48 | 6.95 |
| gamer, libomp, repeat | | 5.79 | 5.17 | 7.13 |
| laptop, libgomp | 15.74 | 13.37 | 13.58 | 11.14 |
| laptop, libomp | | 15.45 | 15.43 | 10.07 |
| xeon one socket, libgomp | | 3.39 | | 2.85 |
| xeon one socket, libomp (noisy, 3.22 to 8.87) | | 6.85 | | 5.36 |
| xeon one socket, libomp, `KMP_BLOCKTIME=0`, median of 3 | | 302.17 | | 142.82 |

| other cases, libomp | 3.3.1 | Fix A | Fix B |
| --- | --- | --- | --- |
| L, gamer | 2.12 | 2.14 | 2.53 |
| kddcup09_churn, gamer | 2.33 | 2.60 | 2.59 |
| susy, gamer | 19.17 | 20.00 | 17.90 |
| L, laptop | 3.43 | 3.32 | 2.26 |
| kddcup09_churn, laptop | 4.05 | 4.08 | 1.58 |
| susy, laptop | 14.03 | 14.08 | 12.06 |

Fix A keeps the gain of 3.3 over 3.2.x on main. The function bodies of `histogram.c` and
`splitting.c` are identical between 3.3.1 and Fix A once blank lines and temporaries are normalized,
since all the HGB loops of main are standalone. So 3.3.1 vs Fix A is close to an A/A test here, and
its gaps of up to ~11% on gamer are noise. I didn't compare the utility code or the other modules.

## Where it doesn't help

- HGB on main with libomp on gamer: Fix B is 23-25% slower on `year_prediction_msd` in 2 rounds, 19%
  on `L`, and all three phases (`hist`, `find_split`, `apply_split`) lose. On the laptop, with the
  same libomp (`llvm-openmp` 23.1.2), Fix B is 35% faster on the same case, and on one socket of the
  Xeon 22% faster. On gamer it also gains on `pr` and `fuse3` (table above). It stops losing when
  threads don't sleep (`KMP_BLOCKTIME=200`: 2.94 s vs 2.54 s) or only E-cores are used (4.34 s vs
  3.43 s). I think deep idle states cause it, but that isn't settled:

  | timer thread on each CPU | C3 entries per fit, 3.3.1 / Fix B | 3.3.1 | Fix B |
  | --- | --- | --- | --- |
  | none | 131k / 359k | 5.15 | 7.09 |
  | 5 ms | 5k / 5k | 4.04 | 3.16 |
  | 200 us | 0 / 0 | 4.01 | 2.79 |

  Two gaps remain: no neutral control yet, since the 5 ms timer already removes C3, and the laptop
  has the same idle table as gamer without the slowdown. Details in `LOG.md`, Exp 9.
- `kddcup09_churn` on gamer: 4-16% slower with the fix, in the 4 configurations. On the Xeon it
  gains. Probably idle states on mixed P and E cores, but I haven't checked.
- The laptop is noisy: its first round showed HGB predict 27% faster and KMeans lloyd 11% slower,
  and neither came back in the second round with 2x the reps.

## Not done

- Nothing was run on Windows, and no scikit-learn build was made with clang.
- The third barrier call site of `with_parallel_two` in `results/local/barriers_generated.txt`
  isn't explained.
