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
