# GEMM experiments log: mac-m2

MacBook Pro M2 (4 P "Avalanche" + 4 E "Blizzard" cores, 16 GB, P-cluster L2
16 MB, E-cluster L2 4 MB), macOS 27.0.1, on AC power, fan-cooled. Libraries:
Accelerate (system), conda-forge OpenBLAS 0.3.34 pthreads and OpenMP builds
(`DYNAMIC_ARCH`, picks the `VORTEX` kernels), conda-forge BLIS 2.1 pthreads.
Remote copy at `~/work/gemm-experiments`; every timed run goes through
`scripts/mac_sweep.sh NAME ...` (benchlock `all`, waits for background CPU
< 40 %, samples `top` every 10 s into `NAME.top`). Summaries with
`pixi run python scripts/mac_summarize.py results/mac-m2/NAME.csv`.
Throughput is GFLOP/s from the median call time, median over rounds.

## Harness changes for macOS (src/gemmbench.c)

- `now()` uses `CLOCK_UPTIME_RAW` on macOS: `CLOCK_MONOTONIC` there has 1 us
  resolution, too coarse for small GEMMs (Accelerate does 64^3 in 2 us).
- `--suffix S` (or env `GEMMBENCH_SUFFIX`) appended to `cblas_?gemm`, to call
  Accelerate's `cblas_dgemm$NEWLAPACK` (macOS 13.3+ interface).
- env `GEMMBENCH_BLAS_THREADING=0|1` calls Accelerate's `BLASSetThreading`
  (macOS 15+, per calling thread: 0 multi-threaded, 1 single-threaded).

## 2026-10-30 smoke: P vs E cores with taskpolicy (not saved)

Purpose: find a way to pin to E cores (macOS has no affinity API).
`taskpolicy -b` (background QoS) confines to E cores but also drops their
clock: OpenBLAS 1T DGEMM 1024^3 7.0 GF/s vs 54.3 on a P core, Accelerate
DGEMM 2048^3 27 vs 309, SGEMM 102 vs 1237. `taskpolicy -c utility` changes
nothing. So E-core-only numbers are only available at background clocks.

## e1_square: square sizes, 4 libraries, T=1,4,8

Purpose: size sweep of Accelerate vs OpenBLAS (pthreads/OpenMP) vs BLIS.
Command: `scripts/mac_sweep.sh e1_square --threads 1,4,8 --variant accelerate
--variant openblas --variant openblas-omp --variant blis --dtype d,s --rounds 3
--shapes 32^3 ... 2048^3` (powers of two).
CSV: `results/mac-m2/e1_square.csv`.
Outcome: Accelerate wins everywhere, by 2-10x for DGEMM and 4-20x for SGEMM.
But OpenBLAS T=4 dropped from 185 GF/s (round 0) to ~102 (rounds 1-2): Spotlight
(`mds`), `FPCKService` and `knowledgeconstructiond` started indexing the new
pixi envs and took 1.5-2.7 cores. Multi-threaded OpenBLAS/BLIS numbers in
this run are unreliable; led to the quiet-wait and `top` sampling in
`mac_sweep.sh`, and to putting the venvs under `venvs.noindex/`.

## e2_threads: thread scaling 1-8, 4 libraries, square + sklearn shapes

Purpose: thread scaling of each library over the 4 P then 4 E cores, on
2048^3, 4096^3 and scikit-learn shapes: `100x100x1000000:NT` (X.T @ X as a
GEMM, X 1e6x100), `10x1000000x100:NN` (X @ W, W 100x10), `8x1000000x100:TN`
and `256x100000x100:TN` (KMeans-like X @ C.T), `20x100000x1000:NN`
(randomized SVD X @ Q), `100x200x784:NN` (MLP minibatch).
Command: `scripts/mac_sweep.sh e2_threads --threads 1,2,3,4,5,6,7,8 --variant
accelerate --variant openblas --variant openblas-omp --variant blis --dtype d,s
--rounds 3 --shapes <the 8 shapes above>`.
CSV: `results/mac-m2/e2_threads.csv` (27 min).
Outcome: too noisy for OpenBLAS/BLIS scaling. Even 1-thread OpenBLAS DGEMM
2048^3 gave 53 / 34 / 41 GF/s in rounds 0/1/2 with best == median within a
round, i.e. the P-cluster clock changed between rounds (DVFS; the battery was
charging 70 -> 97 % and `spotlightknowledged` of the other logged-in user ran
at ~95 % CPU for an hour). Robust results: VECLIB_MAXIMUM_THREADS 1..8 does
not change Accelerate; Accelerate loses on the thin shapes with a small
output dimension (8x1000000x100:TN f64: Accelerate 47 GF/s vs BLIS 8T 111).
Superseded by e8_scaling for the scaling curves.

## e3_accel_controls: Accelerate threading knobs and $NEWLAPACK

Purpose: does any thread control change Accelerate GEMM, and do the macOS
13.3+ `$NEWLAPACK` symbols (what numpy/scipy wheels call) differ from the
legacy `cblas_?gemm`?
Command: `scripts/mac_experiments.sh e3_accel_controls` (variants
`accelerate` = VECLIB_MAXIMUM_THREADS=T, `acc-single` = BLASSetThreading
single-threaded, `acc-newlapack` = `cblas_?gemm$NEWLAPACK`; T=1,8; squares
64..4096, the sklearn shapes and `syrk:100x1000000:N`).
CSV: `results/mac-m2/e3_accel_controls.csv`.
Outcome: all six columns equal within noise (e.g. DGEMM 4096^3 238-246,
SGEMM 1024^3 1089-1095 GF/s). Accelerate's GEMM ignores every thread knob:
it is one AMX stream per call. The rounds stepped down
1360 -> 1095 -> 724 GF/s (SGEMM 1024^3) independently of the variant: P-cluster
clock steps (the AMX unit is clocked with its cluster).

## e4_callers: concurrent single-threaded callers

Purpose: how many AMX units can Accelerate use, and what do the E cores add
to NEON throughput. P threads each loop on their own GEMM for 2 s
(`gemmbench --callers P`); aggregate rate = reps * flops / 2 s.
Command: `scripts/mac_experiments.sh e4_callers` (= `mac_callers.sh 2 rounds,
P=1,2,3,4,5,6,8, d,s, 512^3 and 2048^3`; variants openblas 1 thread,
accelerate with VECLIB_MAXIMUM_THREADS=1, accelerate default).
CSV: `results/mac-m2/e4_callers.csv`.
Outcome (machine already at reduced clocks): Accelerate's aggregate saturates
at 1.0-1.4x one caller (SGEMM 512^3: 892 GF/s with P=1, 1258 with P=2, 1034
with P=8; DGEMM 2048^3: 241 -> 309), so per-caller throughput falls to 1/P:
the P cluster has one AMX unit shared by its cores, and the E cluster's unit
adds little. OpenBLAS 1-thread callers: SGEMM 512^3 78 -> 204 (P=4) -> 221
(P=8) GF/s.

## e8_scaling: thread scaling again, 5 rounds

Purpose: clean thread-scaling curves for OpenBLAS (pthreads) and BLIS, with
Accelerate as reference.
Command: `scripts/mac_experiments.sh e8_scaling` (T=1..8, d and s, 512^3,
2048^3, 100x100x1000000:NT, 10x1000000x100:NN, 8x1000000x100:TN,
256x100000x100:TN, 5 rounds). Table:
`pixi run python scripts/mac_scaling_table.py results/mac-m2/e8_scaling.csv 3,4`.
CSV: `results/mac-m2/e8_scaling.csv`.
Outcome: `spotlightknowledged` (other user, ~98 % of one core) ran during
rounds 0-2 and stopped before round 3 (last seen 11:38:25 in
`e8_scaling.top`), which gives a natural A/B. Clean rounds 3-4: OpenBLAS
DGEMM 512^3 scales 53 -> 181 GF/s at T=4 (3.4x), then falls to 84 at T=5 and
122 at T=8; SGEMM 2048^3 106 -> 364 (T=4) -> 247 (T=8). BLIS behaves the
same. With the one busy background core (rounds 0-2) OpenBLAS T=4 fell to
104-132 GF/s (DGEMM 2048^3 105 vs 185). Accelerate is flat in T.

## e5_small: small matrices

Purpose: where does Accelerate start to win, and what does threading cost
on small GEMMs.
Command: `scripts/mac_experiments.sh e5_small` (accelerate, openblas, blis;
T=1,4,8; d and s; 8^3 .. 384^3 plus 64^3:TN and 100^3:NT; 3 rounds).
CSV: `results/mac-m2/e5_small.csv`. Background was quiet (< 150 % CPU, no
Spotlight).
Outcome: 8^3 and 16^3 are at the 41.7 ns timer tick (unreliable); Accelerate
and OpenBLAS tie there. From 24^3 on Accelerate wins: DGEMM 32^3 112 vs 40
GF/s, 64^3 229 vs 24, 128^3 278 vs 39 (T=1) or 105 (T=4). OpenBLAS 1-thread
NN has a dip exactly at 64^3 (24 GF/s DGEMM, 33 SGEMM; 64^3:TN gives 46 and
96^3 gives 47): power-of-two leading dimension, probably L1 set conflicts.
OpenBLAS T=8 is 25-30 % below T=4 from 128^3 to 384^3 (DGEMM 256^3: 112 vs
152). BLIS with 4 or 8 threads collapses on small sizes (DGEMM 64^3: 13 and
3.6 GF/s vs 38 single-threaded).

## e6_numpy: what numpy users get from the PyPI wheels

Purpose: time the scikit-learn-like matmuls through numpy itself, with the
two numpy 2.5.4 wheels a Mac user can get: `macosx_14_0_arm64` (Accelerate,
calls `cblas_?gemm$NEWLAPACK$ILP64`, and `?syrk` for `X.T @ X`) and
`macosx_11_0_arm64` (bundled scipy-openblas64 0.3.34, 8 threads by default,
`neoversen1` kernels). Venvs in `~/work/gemm-experiments/venvs.noindex/`,
the OpenBLAS one installed from wheels downloaded with
`pip download --platform macosx_13_0_arm64`.
Command: `scripts/mac_experiments.sh e6_numpy` (envs: default, T=4 and T=1
via OPENBLAS_NUM_THREADS/VECLIB_MAXIMUM_THREADS; f64 and f32; 3 rounds).
CSV: `results/mac-m2/e6_numpy.csv`. `mediaanalysisd` used 15-50 % of a core
during part of the run.
Outcome: Accelerate wheel wins on 16 of 18 cases at default settings, by
1.3x (thin X @ W) to 5.6x (SGEMM 512^3: 1369 vs 243 GF/s). OpenBLAS's
default 8 threads hurt: `X.T @ X` (dsyrk) gets 37 GF/s with 8 threads, 38
with 1 and 80 with 4. With OPENBLAS_NUM_THREADS=4, OpenBLAS beats Accelerate
on `X @ C.T` with 8 centers (61 vs 44 GF/s f64, 113 vs 100 f32) and ties on
`X @ W` (64 vs 61 f64).

## e7_thermal: sustained load

Purpose: does the laptop throttle under sustained GEMM load?
Command: `scripts/mac_experiments.sh e7_thermal` (T=4, 2048^3, d and s,
accelerate and openblas interleaved, 12 rounds of >= 5 s per point, 6 min).
CSV: `results/mac-m2/e7_thermal.csv`. Background quiet, on AC, battery full.
Outcome: no thermal drift. Accelerate stays within 225-252 GF/s DGEMM and
911-1050 SGEMM across the 12 rounds. OpenBLAS 4 threads swings 2x without a
trend (SGEMM 366, 359, 362, 162, 188, 157, 295, ...): scheduling, not heat.
With a static split over 4 threads, one thread that loses its P core to
anything (or lands on an E core) stalls the whole call.

## e3b_accel_background: E cores only (taskpolicy -b)

Purpose: E-cluster AMX and NEON throughput. `taskpolicy -b` confines the
process to the E cores but also runs them at background clocks, so this is a
lower bound for what the E cluster can do.
Command: `scripts/mac_experiments.sh e3b_accel_background` (accelerate and
openblas, T=1,4, squares and sklearn shapes, 2 rounds).
CSV: `results/mac-m2/e3b_accel_background.csv`.
Outcome: Accelerate DGEMM 1024^3 27-33 GF/s, SGEMM 99-147; OpenBLAS 7.8 GF/s
on one E core, 40 on four. That is ~10x below the P cluster for both, so a
process started at background QoS (some launchd jobs, `taskpolicy -b`) gets
a ~10x slower BLAS whatever the library.

## e9_kmeans_callers: KMeans-style concurrent small GEMMs

Purpose: scikit-learn's KMeans calls a single-threaded GEMM from each OpenMP
thread on a 256-sample chunk (`X_chunk @ centers.T`, here column-major
`n_clusters x 256 x n_features:TN`). Does Accelerate's single AMX unit
become the bottleneck when 1-8 threads do this at once?
Command: `scripts/mac_experiments.sh e9_kmeans_callers` (`mac_callers.sh`,
3 rounds, P=1,2,4,8 callers, d and s, 1 s windows, shapes 8x256x100:TN,
100x256x100:TN, 1000x256x50:TN).
CSV: `results/mac-m2/e9_kmeans_callers.csv`. Aggregate = reps * flops / 1 s,
or flops * P / median call time when the 20000 reps/thread cap was hit
(small calls; this overstates the rate once callers run on E cores).
Outcome: with small calls Accelerate's aggregate still grows up to 4 callers
(SGEMM 100x256x100: 546 GF/s with 1 caller, 1069 with 2, 1494 with 4, 1519
with 8), then stops; OpenBLAS keeps growing to 8 (87 -> 450). Accelerate stays
ahead at 8 callers for 100 and 1000 clusters (DGEMM 335 vs 192, SGEMM 1519
vs 450), but with 8 clusters OpenBLAS catches up (DGEMM ~240 vs ~190, both
estimates from capped runs).
