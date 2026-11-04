# GEMM findings: OpenBLAS vs MKL on x86 (Xeon socket, Core Ultra)

Most numbers come from one socket of `intel-xeon` (Xeon 6787P, 86 cores, 172
threads, SNC4, slot `s1`), median GF/s, f64 unless noted, inputs first-touched
by one thread. Commands and raw outcomes are in `LOG.md`, CSVs in
`results/xeon-s1/` and `results/gamer/`. Apple M2 findings are in
`FINDINGS-mac.md`. Section 4 covers the level 1/2 and LAPACK routines
scikit-learn calls besides GEMM (exps 20-23).

Shapes are column-major `MxNxK:TT`. In numpy terms: `X.T @ X` is syrk
`d x n`, `X.T @ Y` is `d1 x d2 x n:NT`, `X @ W` is `k x n x d:NN`, `X @ w` is
gemv `T`, `X.T @ r` is gemv `N`.

## Short answers

1. MKL and OpenBLAS have equally fast single-thread kernels (within ~7%). Every
   case where one wins by more than 1.5x comes from threading: how the work
   is split, how many threads are used, and where they run. Five small
   OpenBLAS patches close most of the cases where MKL wins. The MKL side
   needs a fix in the OpenMP runtime that conda-forge ships, not in MKL.
2. Scaling above ~40 cores is mostly not intrinsic. Large GEMMs reach
   ~65-70% of 86 x the single-core speed in both libraries. The bad scaling
   seen in benchmarks comes from four fixable causes: thread placement by
   the OpenMP runtime, OpenBLAS thread-count cliffs, no split over k in
   OpenBLAS, and memory first-touched on one NUMA node. One gap is left
   without a fix: OpenBLAS on mid-size squares at 64-86 threads, where MKL
   is 1.6-5x faster.
3. Outside GEMM, OpenBLAS uses too many threads for small level 1/2 calls
   and for getrf/gesv: they are 2.6-8x slower at 64-86 threads than at their
   best count. Fewer threads in five places (cakedev0/OpenBLAS#8) makes
   scikit-learn 1.2-5x faster end to end at 43-86 threads on the socket
   (Lasso, `solve`, `eigh`), with no slowdown on the laptop, the desktop or
   the M2. Tall QR, Cholesky and `syevd` stay 2-15x behind MKL, and that
   isn't about thread counts.

## 1. Making OpenBLAS as good as MKL

Patches in `patches/`, all on develop 800bb92a, built with the conda-forge
flags. Correctness checks against a naive reference pass for all of them
(`results/xeon-s1/check_all2.txt`; the only failures are a too-strict
threshold at k >= 60000 that also fails with the patches turned off).

| patch | shape | threads | develop | patched | bound MKL |
| --- | --- | ---: | ---: | ---: | ---: |
| split-K for gemm | `X.T @ Y` 100x100x1e6:NT | 86 | 162 | 1760 | 1980 |
| | 200x200x50000:TN | 86 | 680 | 3411 | 4017 |
| | 50x1000x100000:NT, f32 | 86 | 1016 | 3400 | 1334 |
| syrk thread count | `X.T @ X` 990x463810 | 64 | 82 | 974 | 663 |
| | | 86 | 81 | 919 | 453 |
| split-K for syrk | `X.T @ X` 100x1e6 | 86 | 43 | 2096 | 1828 |
| | 300x200000 | 43 | 66 | 2010 | 1917 |
| | 300x200000 | 86 | 66 | 1397 | 2981 |
| gemv split-x | `X.T @ r` 100x200000 | 86 | 21 | 278 | 276 |
| | 126x36168 | 22 | 13 | 154 | 170 |
| | 20x2000000 | 86 | 4.6 | 97 | 94 |
| l1stride | `X @ W` 10x1e6x100 | 22 | 111 | 303 | 290 |
| | 7x116203x100 | 22 | 229 | 343 | 479 |

What each patch does:

- **split-K** (`apply_splitk_gemm.py`, `apply_splitk_syrk.py`): OpenBLAS
  never splits over k, so a small output with a long k (`X.T @ X` with few
  features, `X.T @ Y`) runs on as many threads as there are output tiles.
  When each thread's share of C is below 128x128 and k / threads >= 512,
  the patches split k, run one single-threaded chunk per thread into
  private buffers (64 MiB cap) and reduce them in parallel.
- **syrk thread count** (`apply_syrk_nthreads.py`, one line):
  `level3_syrk_threaded.c` runs single-threaded when n < threads x 16 (16
  is switch_ratio on these targets). With 990 features it drops from 979
  GF/s at 60 threads to 81 at 62. GEMM uses fewer threads in that case. The
  patch does the same.
- **gemv split-x** (`apply_gemv_splitx.py`): gemv N with few rows splits
  rows between threads, so 100 rows on 86 threads gives one row each. The
  patch splits columns and reduces partial y vectors.
- **l1stride** (`apply_l1stride.py`, one line): with one thread along m,
  each thread streams its whole packed copy of B through memory although
  nobody else reads it. It gains 1.3-3x at 8-43 threads but loses 11-27% at
  64-86 threads on thin `X @ W` (and f32 at 43). It needs a guard before
  it is ready.

The split-K and syrk patches would each make a reasonable upstream PR. The
thread-count one is a bug fix.

Where OpenBLAS already wins: bound MKL loses on medium k with a medium
output at 43-86 threads (50x1000x100000:NT f64 86T: MKL 392, develop 755,
patched 986; 1000x1000x100000:NT 86T: 1150 vs 1808), and unbound MKL
over-threads tiny calls (64^3 at 86 threads: MKL 9 GF/s, OpenBLAS 103 on
one thread).

Gaps left, at 64-86 threads, bound MKL vs patched OpenBLAS:

- mid-size squares: 2000x2000x200 f32 10623 vs 2191, 1024^3 f32 5595 vs 1531,
  256^3 f32 2220 vs 430. Binding OpenBLAS's threads (OpenMP build) recovers only
  15-60% of it, so most of it is `level3_thread.c`: its M x N split and the
  synchronisation between threads that share packed B.
- thin `X @ W` (7x116203x100 at 86 threads: MKL 3016 vs ~400).
- `X @ w` in f32 at 64-86 threads (100x200000: MKL ~2100-2590 vs ~300); the
  80 MB matrix fits in the socket's L3.
- small single-threaded calls such as KMeans chunks (10x256x100:TN: MKL 60
  vs 31 on one thread), probably per-call overhead in the small-matrix
  path.

## 2. Making MKL scale: the OpenMP runtime

conda-forge's MKL runs on conda-forge's LLVM libomp (23.1.4). Unbound, which
is the default, it collapses from ~43 threads up, and its speed varies from
process to process for the same setting (43T 4096^3: 1436 or 3227).

| MKL setup, 86 threads | 4096^3 | 7x116203x100 | 100x100x1e6:NT | 100x200000x100 |
| --- | ---: | ---: | ---: | ---: |
| default (LLVM libomp) | 2554 | 1283 | 1781 | 2665 |
| `OMP_PROC_BIND=close OMP_PLACES=cores` | 4758 | 2971 | 1978 | 2663 |
| `KMP_BLOCKTIME=infinite`, unbound | 4607 | 3083 | 3263 | 4998 |
| Intel libiomp5 2026.1 (LD_PRELOAD), unbound | 4787 | 3021 | 3281 | 5024 |

The cause is placement: during the timed calls (20 ms samples of each
thread's CPU), 20-35 of the 86 default threads share a physical core with
another MKL thread while as many cores sit idle, and at 172 threads 35-64
of them are stacked two per CPU (millions of involuntary context switches).
With binding, Intel's runtime or `KMP_BLOCKTIME=infinite`, the threads sit
on distinct cores. The threads do not sleep, migrate more, or make
different syscalls (sched_yield and futex counts are similar between LLVM
and Intel), so I don't know what triggers the bad placement in LLVM's
runtime. A `perf sched` trace of thread creation would be the next step.
GNU libgomp (`MKL_THREADING_LAYER=GNU`, and OpenBLAS's OpenMP build) has the
same problem.

Consequences for threadpoolctl:

- It can't bind MKL's threads from `threadpool_limits`. Binding set after
  the first `mkl_set_num_threads`/`mkl_get_max_threads` call is ignored
  (0 of 86 threads bound), and threadpoolctl makes those calls itself. Set
  right after loading the library, before any call, it works.
- Binding is not a safe default anyway. `close` packing costs 22% with 43
  threads on a socket (it fills the first SNC domains), 1.6-1.9x on
  memory-bound shapes vs Intel's runtime unbound, and two bound processes
  with the same settings pile on the same cores (pair throughput 2935 vs
  5477 on disjoint halves).
- The cleanest fix is upstream: conda-forge's MKL build pulling Intel's
  libiomp5 instead of LLVM libomp, or LLVM libomp placing its workers the
  way Intel's does. For users today, `KMP_BLOCKTIME=infinite` fixes it, at
  the cost of threads that spin forever (bad next to another thread pool,
  see the KMeans contention notes).
- On the hybrid laptop, conda-forge's libomp defaults to `KMP_BLOCKTIME=0`
  (200 ms on the Xeon). On the Xeon, `KMP_BLOCKTIME=0` drops 256^3 at 86
  threads to 20 GF/s.

## 3. Scalability: what is intrinsic

- Large squares scale in both libraries: 16384^3 at 86 threads, memory
  interleaved, OpenBLAS 5386 and MKL 5966 GF/s, i.e. ~63% and ~69% of 86 x
  the one-core speed (~100 GF/s; part of the gap is the lower all-core
  clock, not measured).
- Hyperthreads add nothing for GEMM once 86 threads are well placed: in
  pinned runs 172 threads are equal or slower in every library.
- Memory placement is worth up to ~1.5-1.8x at full-socket thread counts
  when the inputs were written by one thread, as numpy arrays usually are:
  `numactl --interleave` over the 4 SNC nodes gives OpenBLAS 16384^3 86T
  3619 -> 5386 and 100x200000x100 86T 1420 -> 2601. It does not hurt at
  22-43 threads.
- Thin and tall-skinny GEMMs are memory-bound and stop scaling early. That
  part is intrinsic, but MKL's 7x on 7x116203x100 shows OpenBLAS is far
  from that limit.
- Raising OpenBLAS's 128-thread cap (NUM_THREADS=256) brings nothing useful.

Two other things to know:

- conda-forge OpenBLAS 0.3.34 collapses with many concurrent single-threaded
  callers on small calls (KMeans with 10 clusters: 297 GF/s at 22 callers,
  200 at 86, vs 2221 for develop at 86). Something between 0.3.34 and
  develop fixed it. I haven't bisected it.
- On the Core Ultra 9 285 (8P + 16E), OpenBLAS gains nothing reliable from
  the E cores (even split waits for the slower cores) and 24-thread runs
  swing by ±40%. I didn't go further there (see `LOG.md` exp 6).

## 4. Other routines: thread counts

Develop (b8d14a66) vs MKL on Intel's libiomp5, f64, through the Fortran
symbols (`src/gemmbench.c`, `scripts/run_routines.sh`, `routines.csv`).

Where OpenBLAS uses too many threads (86T vs its best count):

- ddot/daxpy thread every call with n > 10000 on every thread. A call costs
  ~0.35 us per thread, so n=1e5 takes 26-34 us at 64-86T vs 4 us at 4-8T.
  The best count is ~n/25000.
- dsymv (inside `eigh`) is best at ~n/125 threads, 2.9-4.3x slower at 86T
  for n = 500-2000.
- getrf/gesv n=1000-4000 is best at 12-16T, 2.6-8x slower at 64-86T (gesv
  2000: 15 ms at 16T, 106 ms at 64T, MKL 8.5 ms).
- trsm with 100 right-hand sides (`cho_solve`) is best at 16-24T, 2.2x slower
  at 86T and 10x behind MKL: each thread packs the whole triangle.

The fix (cakedev0/OpenBLAS#8, `patches/thread_counts_v6.diff`) caps the
count in the interface: ddot/daxpy at max(8, n/25000) threads (axpy in
bytes), dsymv at max(4, n/64), getrf/gesv at max(8, sqrt(m*n)/200). Each cap
only applies while it is below a third of the threads (half for dsymv), so
24 cores or fewer only see the dsymv change. Two reasons for that rule:

- On a hybrid CPU an even split over a few more threads than P cores waits
  for the E cores: getrf 2000 on the Core Ultra 9 285 (8P + 16E) takes 13.5
  ms on 8 threads, 17.5 ms on 10, 16 ms on 24. On the M2, 4 of 8 threads are
  slower than all 8 for dsymv.
- On the Xeon socket, Efficiency Latency Control runs the uncore at full
  speed only when more than 95% of the cores are busy. From 82 of 86 busy
  threads every level 1 call gets 2-2.5x cheaper, so using all the cores
  can beat the "right" count.

Results, develop -> patch:

- One Xeon socket, end to end in scikit-learn at 43-86T: Lasso 20000x100
  3-5x faster, Lasso out of cache 1.5-2.3x, `scipy.linalg.solve` 1000
  2.3-5.4x and 4000 1.3x, `eigh` 1000/2000 1.1-1.3x. Equal at 8-24T.
  Microbenchmarks: getrf 1000 64T 44 -> 9 ms, gesv 2000 86T 77 -> 16 ms,
  dsymv n <= 1000 2-5.7x.
- intel-laptop and intel-gamer-gpu: dsymv n <= 500 1.2-3.2x faster, end to
  end within +-3% (laptop) and +-5% (gamer). M2: same code as develop.
- Given up to stay safe on hybrid CPUs: getrf 8000 at 86T (434 ms vs 290
  with a tighter cap), getrf 4000 at 43T.
- Left: after a call on ~20-30 threads, an all-thread call is up to ~2.2x
  slower for a few hundred ms while the uncore clocks back up (ddot 1e6
  16 -> 36 us). It only shows in back-to-back microbenchmarks.

Not about thread counts, 2-15x behind MKL at every count: tall geqrf
(1e5x200: 116 vs 14 ms), and through it `gelsd` (LinearRegression) and tall
`gesdd` (PCA full SVD); potrf (8000: 100 vs 39 ms); syevd 4000 (1.07 vs 0.46
s). Tracked in cakedev0/OpenBLAS#5, #6 and #7. OpenBLAS wins on tall getrf
(1e5x20: 2.6 vs 6.2 ms) and dger.

Pitfalls when measuring these:

- Lasso with `tol=0` on independent features reaches an exact
  floating-point fixed point and stops early, at an epoch that depends on
  the ddot summation order, so on the thread count. Use correlated features
  and record `n_iter`.
- `LD_PRELOAD` of a custom OpenBLAS into a conda env loads two copies (the
  env's libblas.so.3 still opens through the RPATH, with its own thread
  pool). Swap the library file in a hard-linked copy of the env instead.
- With T threads pinned to exactly T cores, Linux 7.0 sometimes leaves two
  threads on one CPU for seconds, and every call waits ~3 ms (10-40x
  slower). It is random, hits develop too, and needs a mask that cuts a
  sub-NUMA node (cakedev0/OpenBLAS#9). Take the best of several rounds.
- On intel-laptop and intel-gamer-gpu, the first process after an idle
  period runs 15-25% faster. Alternate the order of the variants per round.

## Next steps

Where things stand on the fork (cakedev0/OpenBLAS):

- syrk thread count: PR #3 (v4, measured on the socket, the laptop, the
  desktop and the M2), ready to go upstream.
- thread counts for level 1/2 and getrf/gesv: draft PR #8.
- per-precision SWITCH_RATIO in DYNAMIC_ARCH builds: PR #4, closed. The fix
  is right but slows down small double squares at 43-86T (exp 14).

I think the order should be:

1. Send PR #3 upstream, then PR #8.
2. Propose split-K for gemm and syrk to OpenBLAS with these numbers, after
   tuning the trigger on another core count (only this socket and the
   24-core desktop were measured).
3. Report the LLVM libomp placement problem to conda-forge
   (`notes/issue_intel_repack_llvm_openmp_placement.md`).
4. Look at the mid-size square gap in `level3_thread.c`, the largest GEMM
   gap left and the most work. The LAPACK gaps (#5-#7) come after.
