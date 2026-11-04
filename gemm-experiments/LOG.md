# Experiment log (x86: intel-xeon socket slot, intel-gamer-gpu)

Mac experiments are logged in `LOG-mac.md`. Times are UTC on intel-xeon,
local time elsewhere.

## Setup (2026-10-30)

- conda-forge: libopenblas 0.3.34 (pthreads and openmp builds,
  `MAX_THREADS=128`), mkl 2026.1.0 (Intel threading, llvm-openmp 23.1),
  blis 2.1 (pthreads).
- OpenBLAS 0.3.34 `DYNAMIC_ARCH` picks the **Cooperlake** kernels on the
  Xeon 6787P (Granite Rapids, family 6 model 0xAD): `dynamic.c` has no case
  for extended model 0xA, so it falls through to the generic feature test.
  MKL reports AVX-512 + AMX at 3.77 GHz. On the Core Ultra 9 285 (8 P cores
  = CPUs 0-7, 16 E cores = CPUs 8-23), OpenBLAS picks Haswell and MKL AVX2.
- OpenBLAS `interface/gemm.c` thread heuristic (x86): 1 thread if
  m*n*k <= 65536*4, else min(ncpu, m*n*k / 262144). So 256^3 already asks
  for 64 threads, i.e. ~0.5 MFLOP per thread.
- `notes/sklearn_results_mkl_vs_openblas.md`: where MKL and OpenBLAS differ in
  the existing scikit-learn benchmark results (dsyrk past one NUMA node, thin
  dgemm, dgemv, KMeans' single-threaded sgemm from OpenMP threads).

## Exp 1: MKL vs OpenBLAS vs BLIS map

`scripts/run_map.sh xeon-s1 1,8,86,172` (s1 slot) and
`scripts/run_map.sh gamer 1,8,24` (whole machine). 20 shapes (square
16..4096, sklearn-like tall/skinny), f64 and f32, 2 interleaved rounds,
default library settings (no pinning). Results in `results/*/map.csv`.

Outcome, gamer (`results/gamer/map.csv`, ratios are OpenBLAS/MKL GFLOP/s,
best of 2 rounds of medians):
- 1 thread: parity (0.95-1.05) on everything >= 128, MKL ~20% faster on
  16-64^3 f64 and ~10-15% on thin shapes. The kernels are not the issue.
- 8 and 24 threads: equal on large squares; OpenBLAS wins on mid sizes
  (512^3 f32 at 24T: 1625 vs 578 GF/s, 64^3: MKL threads it and drops to
  39 GF/s vs 137); MKL wins 2-3.4x on small-output/long-k shapes
  (100x100x1e6:NT f64 at 24T: 177 vs 594) and on thin X @ W
  (10x1e6x100 f64: 49 vs 136, same at 8 and 24 threads = no scaling).
- BLIS matches MKL on the thin shapes at 24T: not an intrinsic limit.
- OpenBLAS OpenMP build is clearly worse than pthreads at 24T (hybrid CPU).

Partial Xeon (round 0): MKL anti-scales on squares with default settings
(4096^3 f64 at 86T: 1328 vs OpenBLAS 2749; 64^3: 5.9 GF/s vs 125 at 8T);
OpenBLAS loses 3-12x on small-output/long-k (100x100x1e6:NT at 86T: 79 vs
954) and 2x on thin X @ W.

## Diagnosis from the OpenBLAS sources (develop 800bb92a)

- No split over k: `level3_thread.c` partitions only M and N. For a 100x100
  output it picks nthreads_m = 10 (rows per thread >= SWITCH_RATIO = 8,
  below the 16-row AVX-512 dgemm micro-kernel) and nthreads_n = 2, so at most
  ~20 threads, each on a sliver.
- Thin X @ W (m <= GEMM_P, huge n): the M x N optimizer ends at
  nthreads_m = 1, but `inner_thread` keeps `l1stride = 1` whenever
  nthreads > 1, so each thread writes its whole packed share of B
  (div_n = GEMM_R/2 columns x k, several MB) to a buffer that nobody else
  reads, then reads it back: ~3x the DRAM traffic of the single-threaded path
  (which uses l1stride = 0 and stays in L1). Predicts the observed 2.7x.

## Patches (`patches/`)

- `apply_splitk_gemm.py`: split-K in `interface/gemm.c` when the M x N tiles
  (32x32) can feed fewer than half the threads and k/p >= 512: p
  single-threaded chunks (chunk 0 writes C with beta), scratch <= 64 MiB,
  parallel column-wise reduction. `OPENBLAS_GEMM_SPLITK=0|N` to disable/force.
- `apply_l1stride.py`: `l1stride = 0` also when nthreads_m == 1 (one line).

## Exp 3: MKL threading on a Xeon socket (s1)

`scripts/run_mkl_threading.sh xeon-s1 8,22,43,86,172`, f64, unpinned
process (the slot's 172 CPUs), variants: OMP_PROC_BIND=close / spread with
OMP_PLACES=cores, MKL_DYNAMIC=FALSE, OMP_WAIT_POLICY=passive.

Outcome (`results/xeon-s1/mkl_threading.csv`): up to 43 threads binding
changes little. At 86 and 172 threads, default MKL (no binding, llvm-openmp)
collapses and binding recovers 2-30x: 4096^3 at 86T 1442 -> 4540 (close) /
4731 (spread) GF/s, 256^3 149 -> 1325, 64^3 5.4 -> 80, LR-predict-like
7x116203x100 806 -> 2678, 100x100x1e6 1006 -> 1979. Bound MKL beats
unpinned OpenBLAS by 1.6x on 4096^3 at 86T. Passive wait is a disaster for
fine-grained calls (64^3: 0.3 GF/s). MKL_DYNAMIC=FALSE alone is not the fix.
So the "MKL anti-scales on the Xeon" pattern of the scikit-learn results is
thread migration, not MKL's kernels or partitioning.

## Noise note

Unpinned runs with T < 86 vary up to 2x between processes (threads land on
HT siblings / different SNC domains). From exp 4 on, A/B runs use
`sweep.py --pin compact` (taskset to the first T physical cores, siblings
after all cores are used).

Note: conda-forge's 0.3.34 picks Cooperlake on the Xeon 6787P, my develop
builds pick SapphireRapids; conda-forge and develop results are within +-3%
for s/dgemm, so the kernel table choice is not a factor for these routines.

## Exp 4: OpenBLAS patches A/B (pinned compact)

`scripts/run_patches.sh xeon-s0 8,22,43,86` (s0 slot) and
`scripts/run_patches.sh gamer 8,16,24`; variants: conda-forge 0.3.34,
MKL (no OpenMP binding, but taskset to T cores), develop 800bb92a (base),
split-K v1, l1stride, split-K forced to T chunks. Correctness:
`scripts/check_patches.sh` (all OK; the only "FAIL"s are k = 200000 at
1e-12 relative error, identical with split-K off, i.e. a too-strict
threshold). Results: `results/{xeon-s0,gamer}/patches.csv`.

Outcome:
- Pinned runs are reproducible: conda-forge and base agree within +-3%.
- split-K (v1 trigger) on 100x100x1e6:NT f64: 4.2x at 22T, 6.5x at 43T,
  20x at 86T (88 -> 1742 GF/s, MKL 1732); f32 86T: 133 -> 4080 (MKL 3317).
  200x200x50000:TN at 86T: 5.1x f64 / 8.3x f32 (beats MKL 1.8-2.3x).
  Forced split shows the v1 trigger was too conservative (8T: +1.9x on
  100x100, +1.7x on 200x200x50000) -> v2 trigger: split when
  m*n < T*128^2, k/T >= 512, and >= T/2 chunks fit in 64 MiB scratch.
- l1stride: thin X @ W 1.3-3.5x at 8-43T on the Xeon (10x1e6x100 f64 22T:
  110 -> 302, MKL 290) and on the gamer (7x116203x100 f64 24T: 47 -> 178,
  MKL 102). At 86T on the Xeon it was erratic (one round of 7x116203x100 at
  54 GF/s, f32 thin shapes -30%): re-measured in exp 7 with 3 rounds.
- Gamer: OpenBLAS large squares do not gain from E cores (4096^3 f64:
  8T 569, 24T 568 GF/s; MKL 561 -> 755). Gamer absolute numbers drift
  10-15% between rounds (power/thermal); compare within a round only.
- MKL without OpenMP binding but with taskset to 86 physical cores still
  loses (4096^3: 2531 vs OpenBLAS 3750): migration, not HT siblings.

## Exp 6: hybrid P/E cores (intel-gamer-gpu)

`scripts/run_hybrid.sh gamer`: same sweep under taskset to CPU 0 (P), 8 (E),
0-7 (8P), 8-23 (16E), 0-23 (24), and 8 threads floating on 0-23.
Results `results/gamer/hybrid_*.csv`.

Outcome (f64 4096^3, GF/s): 1 P core 77, 1 E core 50 (sgemm 160 vs 95);
8P 576 (94% of 8x77), 16E 695 (87% of 16x50), all 24: OpenBLAS 864,
MKL 811, BLIS 755 in this run, but 568 vs 755 in exp 4 for the same
setting: at 24 threads on this machine round-to-round swings reach +-40%
(2000x2000x200 f64: OpenBLAS 311 vs the identical-code l1 build 594 in the
same round). 8 threads floating over all 24 CPUs lose 10-15% vs pinned to P.
An even static split over 8P+16E would be E-bound at ~24x43 = 1030 GF/s;
the ideal (P + E throughput) is ~1270.

Build note: develop 800bb92a with `NO_AFFINITY=0 BUILD_BFLOAT16=1` fails with
GCC 15 (`interface/gemm.c:812: passing argument 5 of 'gemm_thread_mn' from
incompatible pointer type` for sbgemm; GCC 15 makes this an error). The
affinity build therefore uses `BUILD_BFLOAT16=0` (bf16 is not benchmarked).

## Exp 7: combined patch (split-K v2 + l1stride) A/B, 3 rounds pinned

`scripts/run_combo.sh` on xeon-s1 (8,22,43,86 T) and gamer (24 T); variants
conda-forge, bound MKL, develop `base`, `l1`, `combo`. `results/*/combo.csv`.

Outcome (median GF/s, max over rounds):
- combo never loses to base beyond noise except thin X @ W at 86T, where the
  l1stride gain disappears (7x116203x100 f64 86T: base 427, l1 410, combo 380;
  f32 43T: 718 -> 534). The l1stride change needs a thread-count guard.
- split-K v2: 100x100x1e6:NT f64 8T 177 -> 336 (MKL 522), 86T 168 -> 1752
  (MKL 1977); f32 86T 210 -> 4080 (MKL 4419). 50x1000x100000:NT now splits too:
  f32 86T 1089 -> 3383, 2.5x bound MKL.
- Gamer 24T: split-K 2.0-2.6x, l1stride 2.8-3.8x on 7-10 output columns
  (7x116203x100 f64 47 -> 177, MKL 130).
- Remaining gap at 86T on the Xeon, untouched by both patches: mid-size
  squares and medium-k shapes. Bound MKL vs combo: 256^3 1351 vs 321,
  1024^3 5612 vs 1629 (f32), 2000x2000x200 5576 vs 1633 (f64), 10554 vs 2245
  (f32). At 22T the same shapes are within 1.0-1.8x. -> exp 10 adds
  bound/unbound OpenMP OpenBLAS on those shapes to separate threading
  overhead from thread placement.

## Exp 8: KMeans-like concurrent single-threaded callers (xeon-s1)

`scripts/run_callers.sh xeon-s1`: P callers each run their own 1-thread GEMM
(the KMeans chunk pattern). `results/xeon-s1/callers.csv`, aggregate GF/s.

Outcome:
- conda-forge OpenBLAS 0.3.34 collapses on the smallest chunk
  (10x256x100:TN, i.e. 10 clusters): f64 297 GF/s at 22 callers, 200 at 86,
  while develop (`base`) scales to 2221 at 86. Larger chunks scale the same
  in both. Something between 0.3.34 and develop 800bb92a fixed a contention
  on small calls (to bisect if it matters for KMeans users on 0.3.34).
- Per call, MKL is 1.2-1.9x faster than OpenBLAS on these small shapes
  (10x256x100:TN f64 1 caller: 60 vs 31; 100x256x784:TN 87 vs 58) and keeps
  that lead at 86 callers (3539 vs 2221). Small-GEMM call overhead and/or
  the TN small-matrix path in OpenBLAS: next thing to profile.

## Exp 9: gemv split-x (gamer done, xeon re-queued)

`scripts/run_gemv.sh`. The first xeon-s1 run lacked the `gemv` build (the
chain stopped when the `aff` build failed its tests, see below).

Gamer outcome: gemv N (X.T @ r) with ~100 rows: 99x26215 f64 24T 5.4 -> 103
GF/s (MKL 63), f32 20.6 -> 207 (MKL 73); 126x36168 f64 8T 12.5 -> 53 (MKL 54).
gemv T unchanged (within noise; one f32 24T case 398 -> 340 to recheck).

Build note: develop `NO_AFFINITY=0 NUM_THREADS=256 BUILD_BFLOAT16=0` builds
but its own BLAS tests segfault (sblat1 at 1 thread, cblat3_3m at 2), so the
`aff` variant is dropped from exp 10.

## Exp 11: MKL OpenMP runtime, binding, late binding (xeon-s1, unpinned)

`scripts/run_mkl_runtime.sh xeon-s1 22,43,86,172`, 2 rounds, unpinned (whole
s1 socket mask, like a user's process). `results/xeon-s1/mkl_runtime.csv`;
the harness now prints how many threads are bound to one core and which
OpenMP runtime is mapped (`.meta.json` "stderr").

f64 median GF/s at 86 threads (bound MKL = `mkl-close`):

| variant | 256^3 | 1024^3 | 4096^3 | 7x116203x100 | 100x100x1e6:NT | 100x200000x100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conda-forge OpenBLAS | 306 | 1461 | 2775 | 351 | 137 | 1226 |
| MKL default (LLVM libomp 23.1.4) | 624 | 1581 | 2554 | 1283 | 1781 | 2665 |
| + OMP_PROC_BIND=close, OMP_PLACES=cores | 1144 | 2496 | 4758 | 2971 | 1978 | 2663 |
| + KMP_BLOCKTIME=infinite (unbound) | 744 | 2174 | 4607 | 3083 | 3263 | 4998 |
| + KMP_BLOCKTIME=0 | 20 | 971 | 2843 | 105 | 1987 | 1590 |
| + KMP_HW_SUBSET=1t | 539 | 2572 | 4707 | 2833 | 1978 | 2398 |
| Intel libiomp5 2026.1.2 (LD_PRELOAD), unbound | 1077 | 2498 | 4787 | 3021 | 3281 | 5024 |
| Intel libiomp5, bound | 1141 | 2458 | 4707 | 2965 | 1981 | 2784 |
| MKL_THREADING_LAYER=GNU, unbound | 617 | 1602 | 2497 | 1356 | 1778 | 2277 |
| GNU, bound | 1147 | 2456 | 4683 | 2934 | 1975 | 2803 |
| bind vars set right after dlopen | 1150 | 2482 | 4717 | 3014 | 1977 | 2879 |
| bind vars set after MKL_Set/Get_Max_Threads | 605 | 1542 | 2493 | 867 | 2302 | 3095 |

Same pattern at 43 and 172 threads (at 172: default 4096^3 1696, bound 4170,
Intel unbound 4207, blocktime=infinite 4265).

Outcome:
- Not (only) migration: Intel's own libiomp5, unbound, is as fast as bound
  MKL on compute-bound shapes and 1.6-1.9x faster than bound on the
  memory-bound ones (100x100x1e6:NT, 100x200000x100). conda-forge's LLVM
  libomp with KMP_BLOCKTIME=infinite does the same. Both runtimes print the
  same defaults with KMP_SETTINGS=1 on the Xeon (KMP_BLOCKTIME=200ms,
  throughput, hyper barriers); Intel's additionally shows
  `KMP_AFFINITY=...,nobind_init,perf_bind`. So LLVM libomp's default waiting
  behaves differently from Intel's at >= 43 threads; exp 11b counts worker
  sleeps (voluntary context switches) to check.
- Binding to cores costs 1.6-1.9x on memory-bound shapes vs the best unbound
  setting (close packing one thread per core over SNC domains filled in
  order vs. the scheduler spreading?), so OMP_PROC_BIND is not a free default.
- Late binding: the bind variables only take effect when set before the
  first MKL thread query. After `MKL_Set_Num_Threads`/`MKL_Get_Max_Threads`
  (what threadpoolctl calls) libomp has read its environment: 0 of 86
  threads bound, same speed as default. A threadpoolctl "bind" feature
  would therefore have to run before anything queries MKL, i.e. it cannot be
  a `threadpool_limits` option.
- On this laptop (hybrid Core Ultra), conda-forge's libomp reports
  KMP_BLOCKTIME=0ms by default (with no OMP_/KMP_ variable set), 200ms on the
  Xeon: the default depends on the CPU (not checked in the libomp sources).

## Exp 11c: two concurrent processes on one socket (xeon-s1)

`scripts/collide.py --threads 43,86`: one shape in two processes started
together, `min-time 2`; total GF/s of the pair (best of 2 rounds).
`results/xeon-s1/collide.csv`.

| T per process | shape | OpenBLAS alone / x2 | MKL default alone / x2 | MKL bound alone / x2 same cores / x2 on socket halves |
| --- | --- | --- | --- | --- |
| 43 | 4096^3 | 3333 / 2848 | 3450 / 3198 | 2690 / 2935 / 5477 |
| 43 | 7x116203x100 | 398 / 543 | 1068 / 612 | 1293 / 728 / 2350 |
| 86 | 4096^3 | 2982 / 3990 | 2916 / 2567 | 4651 / 4191 / 4548 |
| 86 | 7x116203x100 | 349 / 538 | 619 / 598 | 2962 / 1228 / 2359 |

Outcome:
- Two bound processes with the same thread count both bind to the first
  cores of the socket: at 43 threads each, the pair gets about half of what
  the same pair gets on disjoint halves (4096^3: 2935 vs 5477). Binding by
  default is unsafe for multi-process use (joblib/loky workers, several
  jobs on one node) unless every process gets its own mask.
- But two unbound default MKL processes are not better than two bound ones
  (3198 vs 2935; 612 vs 728), and at 86+86 threads bound beats unbound
  (4191 vs 2567).
- `close` binding of 43 threads alone loses 22% on 4096^3 vs unbound (2690 vs
  3450; exp 11 too: 2689 vs Intel unbound 3524): close packs the threads on
  the first SNC domains. `spread` would be the better default for partial
  thread counts.

## Exp 11b: do the workers sleep, migrate or get preempted? (xeon-s1)

`scripts/probe_sleep.sh` (fixed number of calls per process, unpinned; the
harness sums `/proc/self/task/*/status` context switches and
`/proc/self/task/*/sched` `se.nr_migrations`). v1 (`probe_sleep_v1.csv`):
voluntary switches only; v2 (`probe_migrations.csv`): 3 rounds with
involuntary switches and migrations.

Outcome:
- Workers do not sleep in any setting: <= 0.2 voluntary switches per thread
  per call, the same for default, bound, blocktime=infinite and Intel.
- They barely migrate either: 0.1-0.2 migrations per thread per 4096^3 call
  for default and Intel alike. Migration is not the cause (my exp 3 guess
  was wrong).
- Involuntary switches are the difference. Per thread per call, default
  LLVM libomp: 47 (86T, 100x100x1e6), 400 (172T, same), 978 (172T, 4096^3);
  every other setting: < 1. Spinning threads preempted hundreds of times
  per call means two threads of the team share a CPU and yield to each
  other.
- Default speed is bimodal from process to process: 43T 4096^3 1416-1604
  GF/s here vs 3265 in v1 for the same setting; blocktime=infinite, Intel
  and bound stay at 3200-3500.
- `KMP_USE_YIELD=0` makes it worse (172T 256^3: 6 GF/s), consistent with
  stacked threads that now spin a full time slice each.
- `KMP_BLOCKTIME=1000` (1 s) behaves like the default 200 ms, unlike
  `infinite`: the fix is not "spin longer".

Exp 11d (`scripts/probe_placement.sh`, harness `GEMMBENCH_SAMPLE_PLACEMENT=1`)
samples every 20 ms which CPU/core each running thread is on, to confirm the
stacking directly.

## Exp 11d: thread placement (xeon-s1)

`scripts/probe_placement.sh results/xeon-s1/probe_placement.csv 43 86 172`,
3 rounds, 20 ms samples of the CPU of each running thread while timing.
Columns: average running threads, distinct CPUs, distinct physical cores.

| T | shape | setting | GF/s (3 rounds) | running / CPUs / cores |
| --- | --- | --- | --- | --- |
| 43 | 4096^3 | default | 1436, 1439, 3227 | 42.5 / 42.5 / 41.2, 35.5, 42.0 |
| 43 | 4096^3 | Intel libiomp5 | 3527, 3477, 3483 | 43 / 43 / 43 |
| 86 | 4096^3 | default | 2897, 2530, 2585 | 85 / 85 / 50-64 |
| 86 | 4096^3 | bound, Intel, blocktime=inf | 4260-4760 (2 slow rounds) | 84-86 / same / 83-85 |
| 172 | 4096^3 | default | 1758, 1061, 1790 | 168 / 118-132 / 66-69 |
| 172 | 4096^3 | others | 3200-4240 | 168 / 165-169 / 83-85 |
| 172 | 100x100x1e6:NT | default | 752, 406, 429 | 170 / 104-124 / 62-66 |

Outcome: confirmed. With conda-forge's LLVM libomp at default settings the
kernel leaves MKL's spinning workers badly placed for the whole run: at 86
threads 20-35 of them share a physical core with another worker while as
many cores stay idle, and at 172 threads 35-64 are stacked two per CPU
(millions of involuntary switches). With Intel's runtime, blocktime=infinite
or binding, the workers occupy distinct cores. Placement varies from
process to process, hence the bimodal timings. (Intel and blocktime=infinite
also had a few slow rounds with good placement, e.g. 86T 4096^3 2458 and
2769: another, smaller noise source, possibly the s0 neighbour's memory
traffic.)

Exp 11e (`scripts/probe_yield.sh`, strace -c) checks whether default libomp
calls sched_yield while waiting, which would explain why the kernel never
fixes the placement.

## Exp 9 (xeon-s1): gemv split-x A/B

`scripts/run_gemv.sh xeon-s1 1,2,4,8,22,43,86`, 3 rounds pinned.
`results/xeon-s1/gemv.csv`. Checks (`check.txt`): all gemv cases OK.

Outcome: gemv N (X.T @ r) with few rows, f64: 126x36168 22T 13 -> 154 GF/s
(bound MKL 170), 86T 37 -> 199 (MKL 174); 100x200000 86T 21 -> 278 (MKL
276); 20x2000000 86T 4.6 -> 97 (MKL 94); 1000x40000 43T 49 -> 101 (MKL 106).
Wide gemv N (>= 2199 rows) unchanged or slightly better; gemv T: no case
below 0.98x of base (56 cases). Not closed: 20x2000000 at 22-43T (59 vs MKL
94) and 4096x4096 (no split-x there since rows are enough; MKL 1.4-1.7x).

## Exp 11e: syscalls while waiting (xeon-s1, strace -c, 20 calls of 4096^3)

`scripts/probe_yield.sh results/xeon-s1/probe_yield 86 172`.

| setting | T | sched_yield | futex |
| --- | --- | ---: | ---: |
| default LLVM libomp | 86 / 172 | 56932 / 29170 | 644 / 1361 |
| KMP_BLOCKTIME=infinite | 86 / 172 | 136719 / 493598 | 5 / 5 |
| bound (close, cores) | 86 / 172 | 25171 / 21639 | 637 / 1368 |
| Intel libiomp5 | 86 / 172 | 49566 / 57657 | 596 / 1312 |

Outcome: the sched_yield hypothesis is refuted. Every setting yields while
waiting (blocktime=infinite the most), and Intel's runtime makes as many
yield and futex calls as default libomp. The syscall mix does not separate
the good from the bad placement, so the trigger has to be elsewhere: how and
from which thread libomp creates its workers (Intel's runtime reports
`KMP_AFFINITY=...,nobind_init,perf_bind`, which hints at an
initial-placement step that LLVM's lacks) or how the kernel places them at
creation. Not pursued further: `perf sched` traces of thread creation would
be the next step. The practical conclusion does not depend on it.

## Exp 10: high thread counts, memory placement (xeon-s1)

`scripts/run_scaling_hi.sh xeon-s1 22,43,64,86,128,172`, pinned compact, 2
rounds, f64; memory first-touched by one thread (`local`) vs
`numactl --interleave` over the socket's 4 SNC nodes.
`results/xeon-s1/scaling_hi_{local,interleave}.csv`. Variants: conda-forge
OpenBLAS (pthreads), bound MKL, conda-forge OpenBLAS OpenMP build unbound
(`openblas-omp`, libgomp) and bound, develop built with NUM_THREADS=256
(`t256`). (The `aff` NO_AFFINITY=0 build was dropped: it fails its own tests.)

Outcome (GF/s, local memory unless noted):
- syrk cliff: `X.T @ X` 990x463810 runs at 1061 (43T) and then 81 GF/s, one
  core, from 62 threads up (scan: 979 at 60T, 81 at 62T). Cause:
  `level3_syrk_threaded.c` goes single-threaded when n < nthreads *
  switch_ratio (16 here: 990 < 62 x 16). GEMM lowers its thread count
  instead. Bound MKL: 434-750 at 64-172T. -> patch `apply_syrk_nthreads.py`.
- OpenBLAS OpenMP build without binding collapses from 43 threads on mid
  sizes (1024^3: 98 at 64-86T vs 1575-2108 for the pthreads build; 256^3 at
  86T: 5.6). Bound it is close to the pthreads build. Same placement
  problem as MKL's (exp 11), here with libgomp.
- Bound MKL vs OpenBLAS at 86T on mid sizes, both well placed: 2000x2000x200
  5568 vs 1695 (OpenBLAS OpenMP bound: 1946), 1024^3 2471 vs 1575 (2060),
  256^3 1225 vs 283 (464). Thread placement explains a part (bound OpenMP
  OpenBLAS gains 15-60%), most of the gap is OpenBLAS's own partitioning
  and synchronisation in level3_thread.c: not intrinsic, MKL shows 3x more
  is possible on this socket.
- Large squares scale to the end in both: 16384^3 at 86T: OpenBLAS 3619
  (interleaved 5386), MKL 4519 (5966).
- Memory placement: interleaving the socket's 4 SNC nodes helps OpenBLAS on
  large or memory-bound calls at 64-172T when inputs were first-touched by
  one thread (16384^3 86T 3619 -> 5386, 100x200000x100 86T 1420 -> 2601), and
  hurts nothing measurably at 22-43T. numpy arrays are usually written by
  one thread, so `numactl --interleave` (or MPOL_INTERLEAVE in the process)
  is a cheap user-side gain at full-socket thread counts.
- Beyond 86 threads (hyperthreads): no gain for anyone on GEMM; MKL bound
  `close` with 128 threads on 86 cores is erratic (8192^3: 438) because
  some cores get two threads and the static split waits for them.
- NUM_THREADS=256 (t256) vs the conda-forge 128 cap: 172T 4096^3 3957 vs
  3306 interleaved, otherwise within noise. Not worth it.

## Exp 12: all four patches (xeon-s1)

`scripts/run_all.sh xeon-s1 8,22,43,64,86` (develop 800bb92a + split-K v2,
l1stride, gemv split-x, syrk thread count = `builds/all`), 3 rounds pinned.
`results/xeon-s1/all.csv`; checks `check_all.txt`: all OK (218, incl. 40
syrk) except the known k=200000 threshold artifact.

Outcome (all vs develop base; GF/s):
- syrk thread count: 990x463810 f64 64T 82 -> 974, 86T 81 -> 919 (bound MKL
  663 and 453); 1000x1000x50000 86T 83 -> 1016 (MKL 714); f32 likewise
  (64T 165 -> 1621, MKL 1678).
- syrk with few features and long k is the next gap: 300x300x200000 f64
  66 GF/s from 22T up (single-threaded below the cliff too: 300 < 19 x 16),
  282 with the patch, bound MKL 1567-2983. -> syrk split-K
  (`apply_splitk_syrk.py`, exp 13).
- Other patches as in exp 7/9. The l1stride regression on 7x116203x100 at
  64-86T is confirmed (0.75-0.89x; f32 43T 0.73x): needs a guard before
  proposing it.
- Still unpatched gaps at 64-86T: mid-size squares (2000x2000x200 86T: 2191
  vs MKL 10623 in f32), gemv T at 64-86T (100x200000 f32: ~300 vs MKL
  2100-2590, the 80 MB matrix sits in the socket's L3 for MKL).

## Exp 13: syrk split-K (xeon-s1)

`BUILDS="base all all2" scripts/run_all.sh xeon-s1 8,22,43,86 all2`, where
`all2` = all + `apply_splitk_syrk.py`. 3 rounds pinned.
`results/xeon-s1/all2.csv`; checks `check_all2.txt` (442 OK; the 24 FAIL are
syrk 200x60000:T at relerr 1.19e-13 with split-K off, forced and auto alike:
threshold artifact).

Outcome (f64 unless noted, GF/s, all -> all2, bound MKL):
- 100x100x1e6 syrk 86T: 48 -> 2096 (1828); f32 111 -> 4695 (5190).
- 300x300x200000 syrk 43T: 281 -> 2010 (1917); 86T: 234 -> 1397 (2981).
- 50x50x500000:T 86T: 35 -> 1834 (2232).
- 990x463810 and larger n unchanged (no split: n^2 too large for the
  64 MiB cap with enough chunks). Non-syrk shapes unchanged within noise.

FINDINGS.md written from exps 1-13.

## Exp 14: per-precision SWITCH_RATIO in DYNAMIC_ARCH builds (xeon-s1)

In DYNAMIC_ARCH builds the gotoblas table is filled in `kernel/setparam-ref.c`,
compiled without DOUBLE, so `switch_ratio` and `preferred_size` hold the
single-precision values of `param.h` for every precision (16 instead of 8
for d/z on SKYLAKEX, COOPERLAKE, SAPPHIRERAPIDS; also ZEN, HASWELL and
several ARM64 targets). Fork branch `switch-ratio-per-precision` stores both
values and picks by precision. `scripts/xeon_s1_chain5.sh`: four builds from
fork develop b8d14a66 (base, fix, syrk fix, both), checks
(`check_switch_ratio.txt`, 884 OK, nothing unexpected), then
`scripts/run_switch_ratio.sh xeon-s1 8,22,43,64,86`, 3 rounds pinned.
`results/xeon-s1/switch_ratio.csv`.

Outcome (f64, base -> fix, GF/s; f32 unchanged except the usual noise on
256^3/512^3, 0.6-1.2x):
- gains: syrk cliff moves to 124 threads (990x463810 64T 82 -> 905, 86T 82 ->
  898; 1000x50000 ~12x; 500x100000 43T 73 -> 623); 200x200x50000:TN 1.17-1.35x
  at 8-86T; 256x100000x50:TN 86T 1713 -> 2036; with the syrk patch,
  300x200000 syrk another 1.14-1.25x.
- losses: 512^3 43T 939 -> 764, 64T 1261 -> 523, 86T 900 -> 396; 256^3 64-86T
  ~0.5x; 2000x2000x200 86T 1705 -> 1517. Repeated in all rounds and in both
  builds with the fix.
- So switch_ratio 16 was acting as a brake on over-threading small squares
  (the interface hands 512^3 all 86 threads). Static TARGET=SAPPHIRERAPIDS
  builds use 8 for d and probably have these losses today (not measured).
  The fix is right for consistency but not shippable alone: either keep 16
  for double on these x86 targets in param.h, or cap the thread count of
  small squares where it is decided.

## Exp 15: does PR #3 cost anything when syrk calls switch thread counts?

`scripts/xeon_s1_chain6.sh` (OpenMP builds of fork develop and PR #3) then
`scripts/run_syrk_cycle.sh xeon-s1 22,43,86`: each process times every shape
alone, then the shapes called in a loop (`gemmbench --cycle 1`). pthreads,
OpenMP on LLVM libomp (conda-forge's libgomp.so.1), OpenMP on GNU libgomp
(`vendor/gnu-gomp`, from conda-forge `libgomp`). 3 rounds pinned.
`results/xeon-s1/syrk_cycle_{0..4}.csv`, median GF/s.

Outcome:
- Switching thread counts between calls costs nothing measurable, with any
  runtime: n=32 <-> n=160, k=20000 (PR: 2 <-> 10 threads at 86T) cycles at
  136-152 GF/s vs 54-55 on develop, and the PR's cycle time equals the sum
  of the two calls timed alone. k=2000: 118-154 vs 55. n=64 -> 200 -> 990
  (k=2000): 716-835 vs 534-588 at 22-43T, 715-758 vs 80 at 86T.
- Problem found instead: tiny syrk calls get slower. 64x2000 (0.18 ms on one
  thread) runs at 45 GF/s on one thread and 18-26 with the PR's 4 threads;
  32 x {2000, 20000} gains nothing from 2 threads. The cliff kept them on
  one thread before. The PR needs a work-per-thread limit too (exp 16).
- The OpenMP + LLVM + `OMP_PROC_BIND=close OMP_PLACES=cores` variant runs
  every threaded call at 1-12 GF/s, for develop and the PR alike. Probably
  taskset pinning + OMP_PLACES with this build; conda-forge's own OpenMP
  OpenBLAS was fine bound in exp 10. Left out, not investigated.

## Exp 16: small syrk thread-count crossover (xeon-s1)

`scripts/run_syrk_small.sh xeon-s1 1,2,4,8,16,32,64,86`: PR #3 build (threads
= min(T, n/16)), f64 syrk n in {32..512} x k in {1000..100000}, pthreads and
OpenMP on LLVM libomp, 2 rounds pinned. `results/xeon-s1/syrk_small.csv`.

Outcome (both runtimes alike): threading loses below ~2M n^2 k per thread
and gains above ~4M (64x1000 on 4 threads: 47 -> 27 GF/s; 64x5000: 39 -> 60;
128x1000 on 4: 61 -> 129; 256x1000 on 16: 65 -> 290). n=32 gains nothing
from 2 threads (one thread gets all the columns). The gemm interface caps
threads at MNK / 262144; syrk's interface only checks the minimum size.
-> second PR commit: `args.nthreads <= NNK / (8 * SMP_THRESHOLD_MIN *
GEMM_MULTITHREAD_THRESHOLD)` (~3.5M per thread for real types) in
`interface/syrk.c`. Exp 17 re-tests it.

## Exp 17: PR #3 v2 (work-per-thread cap), xeon-s1

`scripts/xeon_s1_chain7.sh`: build `sr-syrk2` (fork branch at 4edf4ccc),
pthreads + OpenMP, checks (`check_syrk2.txt`: 442 OK, only the known
threshold artifacts), `scripts/run_syrk_pr.sh xeon-s1 8,22,43,86` (f64+f32,
3 rounds, `syrk_pr.csv`) and the exp 15 cycles again (`syrk2_cycle_*.csv`).

Outcome:
- f64: the tiny-call loss is gone (64x2000 86T: develop 46, v1 ~21, v2 45-47;
  OpenMP same) and every v1 gain is kept (128x1000 86T 62 -> 129; 990x463810
  86T 82 -> 895; 300x200000 43T 65 -> 283). No f64 case below 0.94x develop.
- Cycles: v2 >= develop in every set and runtime (2 <-> 10 threads,
  n=32 <-> 160, k=20000: 136-151 vs 55; n=64 -> 200 -> 990: 697-800 vs
  538-586 at 22-43T, 716-720 vs 80 at 86T).
- f32: still slower than develop for small n at >= 8 threads: 64x5000
  91 -> 47 (0.51x), 64x{20000,100000} ~0.85x, 128x1000 127 -> 104-108.

## Exp 18: small syrk crossover in f32 and f64 (xeon-s1)

`scripts/run_syrk_small2.sh xeon-s1 1,2,3,4,6,8,16` with the v1 build
(threads = min(T, n/16), no cap), `syrk_small2.csv`.

Outcome: f32 single-thread syrk at small n is 2x faster than f64 (64x5000:
91 vs 39 GF/s), but the threaded driver tops out at the same ~60 GF/s for
n=64 in both, so threads lose in f32 at every k for n=64 (0.5-0.85x), at
k <= 5000 for n=96, at k=1000 for n=128, and with 2 threads for n=192. The
same losses exist on develop at low thread counts (develop with T=4 on n=64
is the v1 run at T=4: f32 64x5000 46 vs 91 on one thread), so they are a
pre-existing small-n inefficiency of threaded syrk that the cliff happened
to hide at high T. 2 or 3 threads on n=64 behave like one thread.

## Exp 19: PR #3 v3 and v4 vs develop (xeon-s1)

v3 (fork branch `syrk-fewer-threads-v3`): v2 + "single precision stays on one
thread below 8 threads" in the interface cap and the driver. `syrk_v3.csv`
(grid n 32..512 x k 1000..100000 + large, f32/f64, 4-86T): still below
develop for f64 96x1000 (4T: 78 -> 54, 8-86T: 57 -> 53; the interface cap left
2 threads, slower than 1) and f32 160x{1000,5000} at 8T (cap -> 1 thread).

v4 (`syrk-fewer-threads-v4`, single commit on develop, driver only): calls
that develop already threads are unchanged; where develop falls back to one
thread (n < nthreads * switch_ratio), use min(n / switch_ratio,
n^2 k / 3.5e6) threads, and one thread if that is < 4 (f64) or < 8 (f32).
`check_syrk4.txt` 442 OK; `syrk_v4.csv`: gains kept (990x463810 86T f64
82 -> 895, f32 166 -> 1519; 300x200000 43T 65 -> 283; f32 128x5000 86T
110 -> 217; f64 128x1000 86T 62 -> 129), former losses back to develop
(f64 96x1000 86T 56 -> 57, 64x2000 46 -> 47; f32 160x1000 138 -> 138). Ten
cases at 0.86-0.95x all sit where v4 runs develop's code path; rerun with
v4 first (`syrk_v4_recheck.csv`) gives 0.97-1.21x: run order and noise.

## Exp 19 (cont.): PR #3 v4 on intel-laptop and mac-m2

`scripts/setup_syrk_v4_remote.sh` (clone fork, build develop and v4 with the
conda-forge flags of each platform, checks, `run_syrk_v3.sh` with
`SMALL_MACHINE=1`, which swaps the two largest shapes for X.T @ X with 8-20
features). intel-laptop: Core Ultra X7 358H, 16 cores, DYNAMIC_ARCH picks
Haswell (switch_ratio 8), 2-16T pinned, 2 rounds. mac-m2: static VORTEX
build (switch_ratio 2, the default), 2-8T unpinned, 3 rounds.
`results/{intel-laptop,mac-m2}/syrk_v4.csv`, checks 442 OK on both.

Outcome:
- laptop: every cell where v4 changes the thread count is a gain (it
  threads: 1.6-2.9x, e.g. f64 96x5000 16T 43 -> 126, f32 96x20000 16T 83 ->
  228, f64 32x20000 8T 29 -> 49) or equal (it keeps one thread: 0.92-1.06x).
  Cells where v4 runs develop's code path spread from 0.65x to 1.2x (median
  0.98-1.0): laptop thermal/noise, not the patch.
- Mac: no change beyond noise. The only changed cells are 8-15 features x
  1e6 samples, now on 4-7 threads, at 0.93-1.04x (memory-bound).
- PR #3 branch now points at the v4 commit (7fb55238); the earlier commits
  stay on fork branch syrk-fewer-threads-v3.
- Thread-switching cycles on v4 (`xeon_s1_chain10.sh`, `syrk4_cycle_*.csv`,
  pthreads + OpenMP LLVM/GNU, 22-86T): v4 >= develop in every set and
  runtime (n=32 <-> 160, k=2000: 132-154 vs 55-56; n=64 -> 200 -> 990:
  705-805 vs 540-590 at 22-43T, 695-728 vs 80 at 86T). Cycle time vs the sum
  of the calls timed alone: median +1.5% (develop +1.2%); worst +27% on the
  shortest cycle (32x2000 <-> 160x2000, 0.37 ms per cycle: waking 10 threads
  after a 1-thread call adds 0.03-0.1 ms), still 2.4x faster than develop.
- PR #3 description rewritten with these numbers.

## Exp 19 (cont.): PR #3 v4 on intel-gamer-gpu

`setup_syrk_v4_remote.sh gamer 2,4,8,12,16,24 PIN=compact ROUNDS=3` (Core
Ultra 9 285, 8P + 16E, DYNAMIC_ARCH Haswell, switch_ratio 8; compact pinning
fills P cores first). `results/gamer/syrk_v4.csv`, checks 442 OK.

Outcome: every cell where v4 threads instead of develop's single thread is a
gain, 1.26-5.08x (f64 160x20000 24T 59 -> 299, f32 160x5000 24T 122 -> 520,
f64 64x20000 12T 48 -> 111), including reduced counts that land on E cores.
Where v4 keeps one thread: 0.955-1.0x+. Cells running develop's code path:
0.76-1.12x, median 0.99 (24-thread noise on this machine, see exp 6).
PR #3 table updated with gamer rows.

## Exp 20: thread scaling of the other routines scikit-learn calls (xeon-s1)

`src/gemmbench.c` now also times float64 level-1/2/3 and LAPACK routines
through their Fortran symbols (`dot:N`, `axpy:N`, `symv:N`, `trsm:MxN`,
`potrf:N`, `getrf:MxN`, `geqrf:MxN`, `syevd:N`, `gesdd:MxN`,
`gelsd:MxNxNRHS`, ...), restoring overwritten inputs outside the timed
region; `--check 1` compares a T-thread call with a 1-thread call.
`scripts/run_routines.sh xeon-s1 1,4,8,16,32,43,64,86`: develop (sr-base,
b8d14a66) vs MKL on Intel's libiomp5, 2 rounds, pinned.
`results/xeon-s1/routines.csv`, table with `scripts/routines_table.py`
(best round's median: one round had ~3 ms stalls on every threaded level-1
call at 86T, another process on the socket).

Outcome, OpenBLAS at 86T vs its best T vs MKL 86T:
- ddot/daxpy n=2e4-1e6: thread from n > 10000 with every thread; the call
  costs ~0.35 us per thread (11 us at 32T, 25 us at 64T), so n=1e5 takes
  26-34 us at 64-86T vs 4 us at 4-8T. MKL: 4-13 us.
- dsymv n=500-2000: best at 8-32T, 2.9-4.3x slower at 86T.
- getrf/gesv n=1000-4000: best at 12-16T, 2.6-8x slower at 64-86T
  (gesv 2000: 15 ms at 16T, 106 ms at 64T; MKL 8.5 ms).
- trsm left with 100 RHS (cho_solve): best at 16-24T, 2.2x slower at 86T,
  MKL 10x faster. Each thread packs the whole triangle.
- No cliff but 3-15x behind MKL: geqrf tall (1e5x200: 116 ms flat from 16T,
  MKL 14 ms), and through it gelsd (LinearRegression, 1e5x100: 79 vs 16 ms)
  and gesdd tall (PCA full SVD, 1e5x100: 116 vs 27 ms); potrf/posv (2000:
  6.4 vs 1.9 ms; 8000: 100 vs 39 ms); syevd 4000 (1.07 s vs 0.46 s).
- OpenBLAS wins: tall getrf (1e5x20: 2.6 vs 6.2 ms), dger.

## Exp 21: fine thread grid (xeon-s1)

`scripts/run_threadgrid.sh xeon-s1`, develop only, `threadgrid-{l12,lapack}.csv`.
Best thread count: ddot/daxpy ~n/25000 (1e5: 4-8, 1e6: 16-24, 5e6: all),
dsymv ~n/125, getrf 4 (n=500) to 24 (n=8000), trsm with 100 RHS ~24
whatever m, potrf flat from 16-24T.

## Exp 22: thread-count patch vs develop (xeon-s1)

`patches/thread_counts.diff` (fork branch thread-counts, built as
`builds/tc`): ddot (x86_64 kernel) and axpy use min(T, n/25000) threads
(axpy: n*sizeof/200000), dsymv min(T, n/128), getrf and gesv at most
max(8, min(m,n)/200). Checks vs one thread at 8/43/86T: 252 OK
(`check_tc.txt`). `scripts/run_tc.sh xeon-s1 4,8,16,32,43,64,86`, 3 rounds,
`tc.csv`.

Outcome (develop -> patch):
- ddot 1e5: 7 -> 4 us at 16T, 26 -> 4 us at 64T; daxpy 2e4 64T 24 -> 4 us;
  dsymv 1000 64T 59 -> 14 us; getrf 1000 64T 45 -> 8 ms; getrf 4000 64T
  158 -> 75 ms; gesv 2000 64T 58 -> 29 ms (86T: 2.4 s -> 24 ms in one
  round); getrf 20000x500 86T 96 -> 29 ms; syevd 1000 86T 73 -> 54 ms.
- Slower, all rounds: ddot/daxpy 2e4-1e5 at 4-8T (patch uses 1-2 threads:
  axpy 2e4 8T 2 -> 4 us), ddot/daxpy 1e6-2e6 at 86T (16-19 -> 19-35 us),
  getrf 20000x500 at 16T (21 -> 27 ms, the column cap is wrong for tall
  matrices), dsymv 4000 at 43T (117 -> 145 us) and 300 at 4-8T (4 -> 5 us).

scikit-learn end to end (`scripts/run_sklearn_e2e.sh xeon-s1 8,43,64,86`,
conda-forge numpy/scipy/sklearn 1.9.1 with the build swapped in by
LD_PRELOAD, checked with LD_DEBUG=bindings; `sklearn_e2e.csv`, best of 3):
Lasso 20000x100 (tol=0, 20 epochs) 69 -> 15 ms at 64T, 37 -> 16 ms at 86T
but 9 -> 15 ms at 8T; Lasso 100000x100 85 -> 46 ms at 64T;
scipy.linalg.solve 1000 62 -> 11 ms at 64T; solve 4000 461 -> 284 ms at
86T; numpy eigh 2000 361 -> 277 ms at 86T. ElasticNet tracks Lasso.
Lasso 1000000x20 is slower at 43-64T (72 -> 92 ms).

Probes (`probe_pool.sh`, `probe_pool.csv`, and inline runs):
- The 86T level-1 results come from the hardware: from 82 busy threads on,
  every level-1 call gets 2-2.5x cheaper (ddot 1e5: 32 us at 81T, 14 us at
  82T; dsymv 1000: 66 -> 53 us), pinned or not, whatever the pool size.
  Socket 1 has Efficiency Latency Control on (elc_high_threshold_enable=1,
  elc_high_threshold_percent=95): the uncore runs at full speed only when
  more than 95% of the cores are busy, and 82/86 = 95.3%. A 40-thread call
  with 43 spinning processes on the other cores gets the same gain (ddot
  1e6 16.5 -> 9.2 us, dsymv 1000 41 -> 30 us, getrf 2000 32 -> 25 ms), 38
  spinners don't. So "all cores busy" is faster on this machine, and
  results at 82-86T compare against a faster uncore.
- Lasso 1000000x20 with develop: 72-76 ms with 43 threads, 70 ms with 24,
  92 ms with exactly 40 (both rounds), on the same 43 cores. A cache-cold
  ddot+daxpy loop over 20 vector pairs of 1e6 is flat at 4.2 ms per cycle
  from 16 to 43 threads, 40 included. Not understood yet.

## Exp 23: thread-count patch v2 (xeon-s1, intel-laptop)

The "40 threads" Lasso 1000000x20 slowdown of exp22 is a benchmark
artifact: with tol=0 and independent features, coordinate descent reaches
an exact floating-point fixed point within a few epochs and stops early,
at an epoch that depends on the ddot summation order, so on the thread
count. One epoch plus the fit overhead is ~70 ms, 20 epochs ~90-100 ms
(`probe_lasso_phases.py`). The coordinate-descent BLAS calls alone (ctypes
loop, `probe_cd_loop.py`) take 20 ms for 20 epochs, flat from 38 to 43
threads. `sklearn_e2e.py` now uses correlated features (all 20 epochs run,
`n_iter` column) and adds a Lasso 100000x1000 case (800 MB, more than the
L3 of every machine; X centered upfront, `check_input=False`).

Patch v2 (`patches/thread_counts_v2.diff`, fork clone branch
`thread-counts`):
- ddot/daxpy: min(T, max(8, n/25000)) threads (axpy: n*sizeof/200000), and
  all T once that count is a third of T. So nothing changes on machines with
  24 cores or fewer, and 86T keeps every core busy for ELC.
- dsymv: min(T, max(4, n/64)); getrf: at most max(8, sqrt(m*n)/200) (tall
  matrices); gesv unchanged from v1 (max(8, n/200)).
v3 (`thread_counts_v3.diff`) compared n/25000 itself with T/3, so it also
capped T = 9-24: Lasso 100000x100 at 16T on intel-laptop went 43 -> 47 ms
and the 800 MB Lasso at 16T on xeon-s1 99 -> 107 ms (8 threads can't stream
from memory as fast as 16). Back to v2. v2 and v3 are identical from 25T up.

Checks vs one thread: all OK but syevr 1000 eigenvectors at 1.07e-10 vs the
1e-10 threshold (develop 1-2e-11, v1 7e-11); dsymv itself is bitwise equal
to one thread, so this is rounding elsewhere in dsyevr amplified by close
eigenvalues.

Results (`tc2.csv`, `tc3.csv` 4 rounds, `sklearn_e2e_tc3.csv` best of 4,
xeon-s1; `tc3.csv`, `sklearn_e2e_tc2.csv` on intel-laptop):
- xeon-s1, 43-86T end to end, develop -> v2: Lasso 20000x100 56-94 -> 13-19
  ms, Lasso 100000x100 68-104 -> 46-49 ms, Lasso 100000x1000 (out of cache)
  183-268 -> 113-119 ms, solve 1000 22-47 -> 9-10 ms, solve 2000 71-117 ->
  51-56 ms, solve 4000 327-425 -> 303-320 ms, eigh 1000 76-96 -> 59-79 ms,
  eigh 2000 343-425 -> 277-397 ms. ElasticNet tracks Lasso. 8-24T: equal.
- xeon-s1 microbenchmarks: same gains as v1 (getrf 1000 64T 44 -> 7 ms,
  gesv 2000 86T 77 -> 15 ms, dsymv 1000 86T 78 -> 22 us, ddot 1e5 64T 26 ->
  4 us), and the v1 regressions are gone (ddot/daxpy at 4-8T, getrf
  20000x500 16T, dsymv 300 and 4000). Left: ddot 1e6 at 86T 16 -> 37 us and
  syevd 200 at 86T 2.6 -> 3.9 ms, both measured right after a shape where
  v2 uses ~20 threads. ELC lag (`probe_stall.sh`): alone, both builds take
  16.5 us and 2.6 ms; after daxpy 5e5 (20 threads) or gesv 4000x100 (20
  threads), v2 takes 36 us and 3.9 ms when measured for 0.2 s, and 16.5 us
  again when measured for 2 s. The uncore takes a few hundred ms to come
  back up after most cores slept. End to end at 86T everything is faster.
- intel-laptop (16 cores, so level 1 unchanged): median of rounds 2-4,
  within +-3% everywhere, solve 1000 and eigh 1000 1.13x faster at 16T.

Measurement problems found on the way:
- Scheduler stalls: with T spinning threads pinned to exactly T CPUs, Linux
  (7.0) sometimes leaves two threads on one CPU for seconds when the mask
  covers part of a sub-NUMA node (T = 24: node 4's 22 cores + 2 of node 5).
  Every threaded call then waits ~3 ms: getrf 8000 at 24T 10x slower, eigh
  2000 40x. Develop too (1 in 4 runs of `probe_stall_seq.sh`, v2 2 in 4).
  A mask of all 86 cores avoids it but spreads threads over nodes (getrf
  2000 20-30% slower); adding the hyperthread siblings
  (`--pin compact-smt`) avoids the 10x stalls but not 2x slowdowns. Kept
  compact pinning and take the best of 4 rounds.
- LD_PRELOAD of a build into a conda env loads two OpenBLAS: numpy/scipy's
  RPATH still opens the env's libblas.so.3 (a second copy with its own
  thread pool, 47 threads at T=24), even though symbols bind to the
  preloaded one. `run_sklearn_e2e.sh` now makes a hard-linked env copy per
  build with the OpenBLAS file replaced.
- intel-laptop: whichever variant runs first after an idle period is
  ~15-25% faster (getrf 8000 at 4T 1.53 vs 2.06 s, identical code).
  `sweep.py` and `run_sklearn_e2e.sh` alternate the variant order per round,
  and `e2e_table.py --median --skip-first` drops round 1.

## Exp 23 (cont.): v2 on intel-gamer-gpu, v4 and v5

intel-gamer-gpu (Core Ultra 9 285, 8 P + 16 E cores), v2 vs develop, 4
rounds (`tc2.csv`, `sklearn_e2e_tc2.csv`): checks OK; level 1 unchanged (24
cores); dsymv n <= 500 1.2-2.3x faster at 12-24T; end to end eigh 1000/2000
1.1-1.2x and solve 1000 1.08-1.19x faster at 16-24T, nothing slower. But
getrf 2000 at 16T 0.82x (10 threads vs 16) and getrf 20000x500 at 24T 0.74x
(15 threads): on a hybrid CPU an even split over a few more threads than P
cores waits for the E-core threads (8 threads 13.5 ms, 10 17.5 ms, 24 16 ms).

v4 (`thread_counts_v4.diff`): getrf/gesv cap only when it at least halves
the count. Fixed 16T on the gamer, but getrf/gesv 2000 at 24T still 0.90x
(10 of 24 threads, 4 rounds consistent). intel-laptop: getrf/gesv within
noise, solve within +-2%.

v5 (`thread_counts_v5.diff`, second commit of cakedev0/OpenBLAS#8): cap
only below a third of the threads, the same rule as dot/axpy. On <= 24
cores only dsymv changes. Every v5 cell is develop's code path or a capped
cell measured with v4. xeon-s1 (`tc5.csv`, `sklearn_e2e_tc5.csv`): checks
OK, getrf 1000 2.9-5.3x and gesv 2000 1.5-4.8x faster from 32T, solve 1000
2.3-5.4x and solve 4000 1.3x at 43-86T, 16-24T equal. Given up vs v4: getrf
4000 at 43T, getrf 8000 at 86T (434 vs 290 ms), small n at exactly 24T.

Scheduler stalls again: in the v4 run, gesv 4000x100 right after gesv 2000
stalled in all 4 rounds at 24T and 32T with v4 (2.0-2.2 s instead of 62-74
ms), and in 1/4 and 2/4 rounds with develop. `probe_stall_gesv.sh` (12 runs
of that pair per build at 24T, compact mask; and at 32T with v5): one stall
per build at 24T (develop in gesv 2000, v4 in gesv 4000 with
OPENBLAS_THREAD_TIMEOUT=30), none at 32T. So random, both builds, not tied
to workers sleeping. Hyperthread siblings in the mask cost 30% on gesv 4000
(535 vs 740 GF/s), so not a fix either.

mac-m2 (M2, 4 P + 4 E cores), v4 vs develop, 4 rounds, unpinned (`tc4.csv`):
only dsymv changes on 8 cores. dsymv 300 at 6-8T 4.2-4.6 -> 2.3 us, but
dsymv 200 at 6-8T 2.8 -> 3.1 us and 500 at 8T 8.7 -> 9.3 us, all rounds.
4 of 8 threads are slower than a 4-thread pool (1.25 us at n=200),
presumably E cores.

v6 (`thread_counts_v6.diff`, third commit of cakedev0/OpenBLAS#8): dsymv
uses all threads once max(4, n/64) reaches half of them. Nothing changes
on 8 cores. Every v6 cell is develop's code path or a v2-v5 capped cell.
- xeon-s1 (`tc6.csv`, `sklearn_e2e_tc6.csv`): dsymv n <= 1000 2-5.7x faster
  from 12-32T up, eigh 1000 1.19-1.33x at 43-86T, eigh 2000 1.14-1.30x at
  64-86T, eigh 4000 within +-2% except 1.05x at 86T. dsymv 4000 at 86T
  177-201 -> 220-226 us with develop's code path: it runs right after dsymv
  2000 on 31 of 86 threads (ELC lag, as in exp23 above).
- intel-laptop: dsymv n <= 500 1.4-3.2x faster at 12-16T, eigh within noise
  (one 57 ms outlier round of 41).
- intel-gamer-gpu: dsymv n <= 500 1.2-2.6x at 12-24T; eigh 1000 over 8
  rounds: 32.0 vs 32.0 ms at 24T, 30.0 vs 31.4 ms at 16T (earlier runs at
  16T: 1.01x, 1.10x), so within +-5%. The 1.1-1.2x eigh gains of the first
  gamer run did not reproduce (develop was slower in that run, 38-44 ms).
- mac-m2 (`tc6.csv`): checks OK; identical code to develop, ratios 0.97-1.03
  except single cells at 0.85-1.20 (noise on this machine).

## Exp 24: split-K as a PR (xeon-s1)

Fork branch `splitk` (22737c10, on develop b8d14a66): `apply_splitk_gemm.py`
and `apply_splitk_syrk.py` without the `OPENBLAS_*_SPLITK` env vars and the
v1 trigger, built as `builds/sk` (`scripts/xeon_s1_chain11.sh`). Checks
(`check_splitk.txt`): all OK but the k = 200000 f64 cases at relerr
1.0e-12 (gemm) and 2.4e-13 (syrk), the same values as with split-K off and
in `check_all2.txt` (threshold artifact), at 3, 8, 43 and 86 threads.

## Exp 25: can PR #4 keep its gains without its losses? (xeon-s1)

Thread scan of develop (`sr-base`), PR #4 (`sr-fix`) and split-K (`sk`),
f64, 8-86T, 2 rounds (`switch_ratio_scan.csv`). GF/s, best round's median:

| shape | T | develop | PR #4 | split-K |
| --- | ---: | ---: | ---: | ---: |
| 200x200x50000:TN | 43 | 751 | 891 | 1918 |
| | 86 | 658 | 890 | 3425 |
| syrk 990x463810 | 86 | 82 | 951 | 82 (PR #3: ~900) |
| syrk 500x100000 | 64 | 72 | 73 | 1408 |
| 256x100000x50:TN | 86 | 1897 | 2058 | (develop's path) |
| 1000^3 | 48 | 1824 | 2037 | (develop's path) |
| 512^3 | 24 | 1088 | 1237 | |
| 512^3 | 64 | 1302 | 551 | |
| 512^3 | 86 | 852 | 392 | |
| 256^3 | 32 | 528 | 187 | |

- The big PR #4 gains are syrk (PR #3 does it, for more shapes) and long-k
  gemm with a small C (split-K does 3.8x better than PR #4).
- What only PR #4 gives: 1.07-1.14x on a few cells (1000^3 and 1024^3 at
  43-48T, 512^3 at 22-24T, 256x100000x50 at 86T).
- Its losses come from the M x N partition in `level3_thread.c`: with
  switch_ratio 8, 512^3 at 64T gets an 8x8 grid of 64x64 blocks on 64
  threads (switch_ratio 16: 32 threads), at 43/86T a 43 x 1 or 43 x 2 grid
  of 12-row strips. No single work-per-thread floor fits both: 256^3 is best
  with ~1M multiply-adds per thread, 512^3 with ~4M.
- So no simple fix. PR #3 + split-K cover the large gains with develop's
  partition for squares; the partition itself is the "mid-size squares" gap.
- syrk 500x100000 at 86T stays at 72 with split-K: the 64 MiB cap allows
  33 chunks, under half of 86 threads, so it falls back to develop's syrk
  (PR #3 fixes that path). Also 24T: 588 vs 1199 at 22T, the 24-thread mask
  stall of exp 23.

## Exp 24 (cont.): split-K v1 vs develop (xeon-s1)

`scripts/run_splitk.sh xeon-s1 8,16,24,43,64,86` (`splitk.csv`, 3 rounds,
f32/f64; gemm `m x m x k:NT` and syrk `m x k:N` for m = 32-1100, k =
5000-1e6, plus a few others). develop / split-K time, min of rounds:
- Gains wherever it splits and C is small: syrk with 32-300 features and
  k >= 20000 8-60x from 16T up (100x1e6:T 86T 41x), gemm 32x32 to 128x128
  with long k 5-40x, 200x200x50000:TN 5.2x (f32 7.9x), 10x200x1e6:NT 4.6x.
- Unchanged where it doesn't split (800-1100 and 2000^3 within noise,
  24T cells excepted: the 24T mask stall hits either build).
- Slower: 512x512x20000:NT d at 24-64T 0.49-0.74x (f32 0.56-0.60x at
  43-64T), 512x512x50000 d 86T 0.83x, f32 800x800x20000 syrk 43T 0.62x,
  256x256x5000:NT d 16T 0.73x.

Cause of the large-C slowdowns: every slow cell needs more than 32 MiB of
partial products ((p - 1) * m * n * size), glibc's largest mmap threshold,
so `malloc` maps fresh pages on every call and the workers fault them in
again. 256x256x50000 d: 63 buffers (31 MiB) at 64T 4x faster, 85 (42 MiB) at
86T 0.98x. With `MALLOC_MMAP_THRESHOLD_=2e8 MALLOC_TRIM_THRESHOLD_=4e8`
(heap reuse), split-K 512x512x20000 43T goes 683 -> 1554 GF/s (develop
1268) and 256x256x50000 86T 644 -> 2893 (develop 669).

256x256x5000 at 16T is a different case: 9 chunks (k / 512) on 16 threads,
while develop's 16-thread M x N split already gives each thread 64 x 64.

v2 (fork `splitk` 7b5e80a7): partial products capped at 24 MiB, and with
fewer chunks than threads, split only if M x N would give each thread less
than 64 x 64 of C.

## Exp 26: split-K v2 vs develop (xeon-s1, intel-laptop)

Same grid as exp 24 (`sk2.csv`, `check_sk2.txt`; `scripts/xeon_s1_chain12.sh`,
`SK=sk2 scripts/setup_splitk_remote.sh`). Checks: all OK but the f64 threshold
artifacts (100x100x200000:NT, syrk 200x60000:T), which fail the same with
split-K off. The checks' `splitk=0/7` rows run the automatic rule: the
`OPENBLAS_*_SPLITK` env vars only exist in the exp 17 prototype.

xeon-s1 (8-86T): every v1 slowdown is gone (512x512x20000 d 0.98-1.00, f32
syrk 800 43T 0.97) and the gains are kept (same table as v1 within noise).
The 10 cells under 0.95 (0.85-0.94, mostly 64-86T, k = 5000) all have one
chunk, i.e. develop's code path; v1 had the same cells on the other side.
Missed gains: at 86T, 32-128 x 20000 falls back to develop (k / 86 < 512 gives
39 chunks, under half the threads) while 64T gets 4-20x.

intel-laptop (2-16T; 4 P + 8 E + 4 LP-E cores, no SMT): gains 1.1-7x on small
C (syrk 64-100 features 5-7x at 12-16T, gemm 32x32 2-3.6x). One real loss:
128x128x5000:NT at 12-16T, d 0.77-0.78x, f32 0.81-0.91x: 9 chunks (k / 512)
on 12-16 threads. A static equal split runs at the pace of the slowest core
(an E core here), which gets 1/9 of the work instead of develop's 1/12: 0.75x
predicted. Other cells under 0.95 have one chunk (noise; the laptop's first
round is often faster, and 2000^3 16T 0.65x on both dtypes is one-chunk too).

v3 (fork `splitk-v3` 1cfb856e): when C is tiny (M x N would give each thread
less than 64 x 64), accept chunks down to 256 so there is one chunk per
thread. A/B of develop, v2 and v3 on the shapes where the chunk count
changes: `scripts/xeon_s1_chain13.sh`, laptop `SK=sk3 BRANCH=splitk-v3
setup_splitk_remote.sh` (`sk3.csv`).

## Exp 27: split-K v3 vs v2 vs develop (xeon-s1, intel-laptop)

`sk3.csv` (4 rounds, gemm `m x m x k:NT` and syrk `m x k:N`, m = 32-512,
k = 5000-50000). Checks `check_sk3.txt`: same two f64 threshold artifacts only.

- intel-laptop: fixes 128x128x5000:NT at 12-16T (d 0.85-0.88x with v2,
  1.02-1.14x with v3; f32 1.24-1.33x), per round v3 0.60-0.65 ms vs develop
  0.66-0.73 vs v2 0.78-0.83. The 4T column at 0.77-0.89x for every shape,
  512x512 included, is one fast develop round (11.7 ms vs ~14.2 in all other
  rounds of all builds). 2000^3 recheck (`sk3_2000.csv`, 6 rounds, 8 and
  16T): the three builds match round for round; the laptop slows ~1.5x after
  3 rounds, which made v2's 2000^3 16T look like 0.65x.
- xeon-s1: new gains where v2 fell back to develop: 32x32x5000 at 24-86T
  6-34x, 128x128x5000 16T 1.31 -> 2.05x, 128x128x20000 64T 19 -> 24x (syrk).
  Smaller gain on 32x32x20000 at 64-86T (64-86 chunks of 232-312 instead of
  39 of 512): 9-11x instead of ~14x. Cells under 0.95x (s 64x64x5000 24-64T
  0.56-0.86 for v2 and v3, d 64x64x20000 86T 0.92, s syrk 32x20000 86T 0.90)
  take develop's path in both builds; s 64x64x5000 varies 2x within a build
  (0.23-0.56 ms). 10-round recheck queued (`sk3_recheck.csv`).

Fork `splitk` fast-forwarded to v3 (1cfb856e); draft PR cakedev0/OpenBLAS#10.

## Exp 28: split-K v2 on intel-gamer-gpu, develop's path slower

`sk2.csv` (4-24T, full grid). Checks: the two f64 threshold artifacts only.
Gains where it splits, 24T: syrk 32-100 columns 8-12x, gemm 64x64 to 200x200
C 1.5-2.3x; 128x128x5000 at 16T 0.87-0.96 (9 chunks, the laptop case, fixed
in v3).

But one-chunk cells (develop's code) are 5-35% slower at 12-24T, the same in
every round with the order alternating: 2000^3 24T 18.2 vs 27.8 ms (f32 9.1
vs 14.8), 1000x1000x100000 24T 0.79x, 512-1100 square 0.85-0.95x. None at
4-8T. Same base commit, gcc 15.3.0-7 and flags for both builds (sr-base built
the day before). `dgemm_`'s frame is 0xc8 vs 0xb0 bytes: `gemm_splitk` is
not inlined and returns before using its arrays. Not seen on xeon-s1 or the
laptop. `perf` is not allowed there (perf_event_paranoid 4).

Queued `scripts/gamer_splitk_layout.sh`: develop, v2, v3 and v3 with the
split-K calls unreachable (`sk3off`, same code), 6 rounds on the slow shapes
(`sk3_layout.csv`). sk3off slow = code layout; sk3off fast = the call costs.

Result (`sk3_layout.csv`, 6 rounds, 2026-11-04 19:55-20:03 UTC): the gap is
gone. On the four shapes at 8-24T, the medians over rounds of develop, v2,
v3 and sk3off are within ~3% of each other, and the mins move both ways.
develop itself is now as slow as the split-K builds were: 2000^3 24T f64
24-29 ms per round (18 ms in `sk2.csv`), f32 13-15 ms (9 ms). So it's not
the code layout or the call; the fast develop rounds of `sk2.csv` were a
machine state I can't reproduce. Another agent runs idle-state and timer
experiments on this host between my runs, which may be related. The one
cell still lower, d 512x512x20000 24T (v2/v3 median 21.0 vs develop 18.9 ms,
no split: the 24 MiB cap gives 13 chunks), is bimodal for develop and sk3off
too (15-22 ms). Split cells gain as before (s 512x512x20000 24T 1.2-1.3x).

Queued a full grid of v3 vs develop, 4 rounds at 4-24T (`sk3.csv`).

## Exp 29: split-K v2 on mac-m2

`sk2.csv` (2-8T, full grid, 3 rounds; threads aren't pinned on macOS).
Checks: the two f64 threshold artifacts only. Gains where it splits: 1.2-3x
(32x32 C 1.2-3.2x, 100-256 columns 1.5-2.5x at 6-8T). 48 of 496 cells have
a median under 0.95x, nearly all at 6T, where 4 P + 2 E cores run and
develop's rounds are bimodal. Slower by both min and median: one-chunk
cells (d 1100x1100x5000 6T 0.81, d 512x512x5000 6T 0.91, s syrk 800x5000 8T
0.91), which run develop's code, and d syrk 64x5000 and 64x20000 at 6T
(6 chunks, 0.81-0.84; 1.03-1.48x at 2, 4 and 8T). v3 splits those the same
way. Running v3 vs develop with 5 rounds (`sk3.csv`).

Full grid, v3 vs develop, 4 rounds (`sk3.csv`, 20:08-20:40 UTC): the gap is
back, the same in all 4 rounds. 2000^3 24T f64 18.2 vs 27.3 ms, f32 9.3 vs
14.4 ms, 1000x1000x100000 24T 0.81-0.83x, 256x256x5000 24T 0.59x; none at
4-8T. So `sk3_layout.csv` only hid it: there develop was slow too. One
difference between the two runs: each gemmbench process times all its
shapes in sequence, and in the grid many split shapes run before 2000^3.
Suspect: split-K's freed buffers (up to 24 MiB) raise glibc's dynamic mmap
threshold, so the later matrices come from the heap instead of fresh mmaps.
`scripts/gamer_splitk_malloc.sh` checks the slow shapes alone in their
process (`sk3_alone.csv`) and the grid with the threshold fixed at 128 KiB
(`sk3_mmapfix.csv`).
