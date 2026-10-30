# GEMM findings: mac-m2 (Apple M2, Accelerate vs OpenBLAS vs BLIS)

MacBook Pro M2: 4 P cores (3.49 GHz, NEON peak 55.8 GFLOP/s f64 and 112 f32 per core) and 4 E cores. The P cluster has one AMX unit shared by its cores. macOS 27.0.1 with the system Accelerate, conda-forge OpenBLAS 0.3.34 (pthreads and OpenMP builds, `VORTEX` kernels, which are the Neoverse N1 NEON kernels) and BLIS 2.1. All numbers are GFLOP/s from the median call time. Commands are in `LOG-mac.md` and CSVs in `results/mac-m2/`.

On this laptop, P-cluster clock changes and scheduling move numbers by up to 2x between rounds (another logged-in user's Spotlight daemon, battery charging). Single-threaded and Accelerate numbers are stable once the machine is quiet. Multi-threaded OpenBLAS and BLIS numbers come from the quiet rounds 3-4 of `e8_scaling`.

## Where Accelerate wins and where it loses

One Accelerate call reaches 347 (f64) and 1376 (f32) at 512^3. That is 6.5x and 13x one P core with OpenBLAS (53 and 106, 95% of NEON peak), and 1.9x and 3.8x the best OpenBLAS result on 4 P cores (181 and 364).

| shape (column-major) | dtype | Accelerate | OpenBLAS 1T | OpenBLAS 4T | BLIS 4T |
| --- | --- | ---: | ---: | ---: | ---: |
| 64^3 | f64 | 229 | 24 | 24 | 13 |
| 128^3 | f64 | 278 | 39 | 105 | 55 |
| 512^3 | f64 | 347 | 53 | 181 | 156 |
| 2048^3 | f64 | 268 | 53 | 172-185 | 156-167 |
| 2048^3 | f32 | 1039 | 106 | 364-371 | 309-330 |
| X.T @ X, 100x100x1e6:NT | f64 | 176 | 43 | 101 | 96 |
| X.T @ X, 100x100x1e6:NT | f32 | 409 | 81 | 195 | 162 |
| X @ W, 10x1e6x100:NN | f64 | 75 | 27 | 61-72 | 84-90 |
| X @ W, 10x1e6x100:NN | f32 | 145 | 30 | 96-104 | 143-159 |
| X @ C.T, 8x1e6x100:TN | f64 | 48 | 34 | 70 | 111-116 |
| X @ C.T, 8x1e6x100:TN | f32 | 120 | 38 | 107-121 | 182 |
| X @ C.T, 256x1e5x100:TN | f64 | 280 | 50 | 160 | 155 |

Accelerate wins from 24^3 up (below that both are at the timer tick) and on every shape with a large enough output. It loses only on tall-skinny GEMMs with 8-10 output columns, where BLIS on 4 P cores is up to 2.3x faster. Those shapes are memory bound: BLIS reads X at about 60 GB/s with 4 cores, while Accelerate does the call as one AMX stream at about 26 GB/s. The AMX tile is 8x8 in f64 and 16x16 in f32, so 8-10 output columns also leave most of it idle.

Accelerate also gets slower on large squares: f32 1376 at 512^3, 1039 at 2048^3, about 830 at 4096^3, and f64 347, 268, about 245. OpenBLAS stays flat across those sizes. The cause is unknown; one guess is blocking for the 16 MB L2 that the AMX unit shares.

Through numpy (`e6_numpy`, numpy 2.5.4 wheels at default settings), the Accelerate wheel wins 16 of 18 cases, by 1.3x (X @ W) to 5.6x (f32 512^3). The biggest gap is `X.T @ X`, which numpy sends to `?syrk`: 142 with Accelerate vs 37 with the OpenBLAS wheel. With its default 8 threads the OpenBLAS wheel is at its 1-thread speed (38); with 4 threads it gets 80.

## Thread controls on Accelerate do nothing for GEMM

`VECLIB_MAXIMUM_THREADS` from 1 to 8, `BLASSetThreading` (macOS 15+) and the `$NEWLAPACK` symbols (macOS 13.3+, which the numpy and scipy wheels call) give the same numbers within noise. `threadpoolctl` does not see Accelerate (`threadpool_info()` is empty), so scikit-learn's `threadpool_limits` cannot control it, and does not need to.

Concurrent callers (`e4_callers`) saturate at 1.0-1.4x one caller on large calls (f32 512^3: 892 with 1 caller, 1258 with 2, 1034 with 8). The 4 P cores share one AMX unit, and the E cluster's unit adds little. On KMeans-sized chunks (`e9_kmeans_callers`) the total grows to about 2.7x at 4 callers (f32 with 100 clusters: 546 to 1494) because per-call overhead overlaps, then stops. Accelerate is still 1.7-3.4x faster than 8 OpenBLAS callers there, except with 8 clusters where they are on par.

So on M2 the Accelerate limit is intrinsic: one AMX unit per P cluster, about 347 GFLOP/s f64 and 1.4 TFLOP/s f32. M2 Pro, Max and Ultra chips have more P clusters, so more AMX units. Whether Accelerate splits one call across them is untested.

## OpenBLAS and BLIS stop scaling at 4 threads on 4P + 4E

Speed-up vs 1 thread, quiet rounds of `e8_scaling`:

| | T=2 | T=3 | T=4 | T=5 | T=6 | T=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenBLAS f64 512^3 | 1.9 | 2.5 | 3.4 | 1.6 | 1.9 | 2.3 |
| OpenBLAS f32 2048^3 | 1.8 | 2.6 | 3.4 | 2.5 | 2.8 | 2.3 |
| BLIS f32 2048^3 | 1.9 | 2.7 | 3.0 | 2.9 | 3.0 | 2.7 |
| OpenBLAS f64 X.T @ X | 1.5 | 2.0 | 2.3 | 1.3 | 1.7 | 1.2 |

Up to 4 threads it scales to 3.4x on squares. The 5th thread lands on an E core. `level3_thread.c` splits M and N evenly, so each call waits for that slowest thread: f64 512^3 drops from 181 to 84 at T=5. That puts an E core at about 17 GFLOP/s f64 (84 / 5), against 53 for a P core.

OpenBLAS takes its default thread count from `hw.physicalcpu` (8), so by default it runs in this bad regime. It is 15-50% slower than 4 threads from 128^3 up (f32 2048^3: 247 vs 364; f64 256^3: 112 vs 152) and 2x slower on X.T @ X. Small sizes are not affected, since OpenBLAS stays single-threaded up to 64^3. BLIS with 4 or 8 threads collapses on small sizes (f64 64^3: 13 and 3.6, vs 38 on one thread).

Even at 4 threads the even split is fragile. Under 6 minutes of sustained load, Accelerate stays within ±7%, while 4-thread OpenBLAS swings between 157 and 366 (f32 2048^3) with no trend. That is scheduling, not heat. One background process using a full core took f64 2048^3 at 4 threads from 185 to 105. macOS has no thread affinity, so a library cannot pin its threads to P cores.

This limit is not intrinsic. Splitting work by core speed should give about 181 + 4 × 17 ≈ 250 for f64 at 512^3 (estimate, not tested): about 35% more than 4 threads and 2x the current 8-thread result. Tall-skinny shapes are different: they stop at 2-2.5x on 4 cores (f64 X @ W: 27 to 61-72) because they are memory bound.

## Suggested changes

In OpenBLAS on Apple Silicon:
1. Default the thread count to the number of P cores (`hw.perflevel0.physicalcpu`) instead of `hw.physicalcpu`, in `get_num_procs` (`driver/others/memory.c`). That is a one-line change worth 15-50% on medium and large GEMMs and 2x on `?syrk` with the defaults numpy and conda-forge users get.
2. Better: split M and N by core speed, or hand out blocks dynamically in `level3_thread.c`. Then the E cores add about 35% instead of costing it, and background load on a P core hurts less.
3. Look at the 1-thread NN dip at 64^3. In f64 it gets 24, vs 46 for 64^3:TN and 47 for 96^3; in f32, 33 vs 89 and 96. This is likely the same power-of-two L1 set-conflict problem as on x86.
4. Using AMX directly is not realistic on M1-M3, since its instructions are undocumented. The M4 has SME, which OpenBLAS already supports with the `VORTEXM4` target. Forwarding calls from OpenBLAS to Accelerate makes less sense than choosing Accelerate at install time.

In Accelerate (for Apple): split memory-bound tall-skinny GEMMs across cores with NEON instead of one AMX stream (BLIS is 1.5-2.3x faster on them), and look at the slowdown on large squares.

## How numpy and scikit-learn users on macOS should configure BLAS

What gets installed today:
- pip, macOS 14+: numpy 2.5.4 (`macosx_14_0_arm64`) and scipy 1.18.1 use Accelerate through the `$NEWLAPACK` symbols (numpy ILP64, scipy LP64). scikit-learn's wheel has no BLAS of its own (it bundles `libomp`) and calls scipy's BLAS from Cython, so it uses Accelerate too.
- pip, macOS 11-13: numpy bundles scipy-openblas64 0.3.34 and scipy bundles scipy-openblas 0.3.31.dev, so there are two OpenBLAS pools of 8 threads each.
- conda-forge: `libblas` defaults to OpenBLAS. `libblas=*=*newaccelerate` and `*blis` builds exist; MKL does not exist on osx-arm64.

Recommendations:
- Use Accelerate. The pip wheels on macOS 14+ already do; on conda-forge, install `"libblas=*=*newaccelerate"`. Its thread settings have no effect on GEMM, so leave them alone.
- If you stay on OpenBLAS, set `OPENBLAS_NUM_THREADS` to the number of P cores (`sysctl -n hw.perflevel0.physicalcpu`).
- The exception is GEMMs on large data with very few output columns (X @ W with about 10 outputs, KMeans with fewer than about 10 clusters), where OpenBLAS or BLIS on the P cores is 1.1-2.3x faster. These calls are fast anyway, so they are not a reason to keep OpenBLAS.
- Do not run BLAS work at background QoS (`taskpolicy -b`, some launchd jobs). It confines the process to slowed-down E cores and every library gets about 10x slower (Accelerate f64 1024^3: about 30 vs 300).

Open questions: does Accelerate use both AMX units of an M2 Pro or Max for one call, and does a speed-weighted split in OpenBLAS actually reach about 250? The second needs an OpenBLAS build on the Mac.
