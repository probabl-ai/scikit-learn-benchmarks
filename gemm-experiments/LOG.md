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
