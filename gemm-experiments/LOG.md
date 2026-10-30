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
