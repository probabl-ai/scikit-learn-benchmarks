# MKL vs OpenBLAS in the existing scikit-learn benchmark results

Mined from `results/` on `main` (2026-10-30) by pairing records with
`read_all_results` + `minimal_match_key` (same hardware/case/env, latest
result per env). Times are medians in ms. Ratio = OpenBLAS time / MKL time.
Routine attributions come from reading the scikit-learn code paths, not from
profiling.

Only two machines have MKL runs: intel-xeon (Xeon 6787P, 2x86 cores,
8 NUMA nodes) and intel-laptop (Core Ultra X7 358H, 16 cores P/E/LP-E).

Versions: MKL 2026.1.0 (Intel threading layer on llvm-openmp 23.1),
conda-forge libopenblas 0.3.34 pthreads and openmp (Cooperlake kernels on the
Xeon, Haswell on the laptop), PyPI scipy-openblas 0.3.34.106.0 plus scipy
1.18.1's vendored 0.3.31.dev.

Caveat: `all_models.py` sets no thread count, so each library uses its
default: on the Xeon MKL 172, conda-forge OpenBLAS 128 (`MAX_THREADS=128`),
PyPI OpenBLAS 64. Only `models_scalability.py` pins equal core counts, and
only for MKL vs PyPI.

## Xeon, MKL wins

| Case | Routine (guess) | Numbers |
|---|---|---|
| Ridge cholesky, year_prediction_msd 463810x990 f64 | `dsyrk` 990^2 k=463810, then `potrf` | 16 cores 2453 vs 2350 (0.96); 32 cores 7418 vs 2551 (2.91x); ~2.0x at 64/128/172. PyPI OpenBLAS jumps 2.35 s -> 7.4 s exactly when the run spans > 1 NUMA node (~21 cores) |
| LogisticRegression lbfgs, covtype+Nystroem 200000x100, 7 classes (vs PyPI) | thin `dgemm` 200000x100 @ 100x7 | 2 cores 4.69x, 4 cores 5.01x (scipy 0.3.31 dtrtrs threading bug), 172 cores 1.61x, 1 CPU 1.29x |
| LogisticRegression lbfgs, bank_marketing 36168x126 | 2x `dgemv` per iteration | 445.7 (omp) / 336.9 (pthreads) vs 109.1: 3-4x, same iteration count |
| LinearRegression wide 1100x8900, 2200x17900 | `gelsd` (LAPACK) | 3.20x, 2.62x |
| LogisticRegression newton-cholesky 800000x100 | `dgemm` p x n x p (sandwich_dot) + p^2 solve | 2.82x (8Mx20 F: 1.94x) |
| Ridge ames_housing 1168x25165 | `dsyrk` 1168^2 k=25165 | 2.48x |
| Ridge synthetic f32 50000x1000 F | `ssyrk` 1000^2 k=50000 | 2.20x |

## Xeon, OpenBLAS wins

| Case | Routine (guess) | Numbers |
|---|---|---|
| KMeans k=1000 sift 100000x128 f32, OMP_NUM_THREADS=64 | single-threaded `sgemm` 256x1000x128 per chunk, from each OpenMP thread | MKL 55728 vs PyPI 27222 (2.05x), omp 39404 (1.41x); only 1.17x at 128 threads |
| KMeans k=100 fashion_mnist 60000x784, 64 threads | `sgemm` 256x100x784 | 1.4-1.9x per iteration; even at 128 threads |
| LogisticRegression predict covtype 116203x100 (vs PyPI) | thin `dgemm` 116203x100 @ 100x7 | MKL anti-scales: 16 cores 13.6 vs 13.1, 32: 25.8 vs 12.5, 64: 31.7 vs 10.3 (3.1x), 172: 30.5 vs 13.5 |

## Core Ultra laptop

MKL wins: LinearRegression `gelsd` 3.1-3.4x, tall `sgeqrf` 2.3x;
LogisticRegression lbfgs `dgemv` (amazon 26215x99) 3.16x vs omp, 22x vs PyPI
(dtrtrs bug); thin `dgemm` 20000x1000x5 2.53x; Ridge `dsyrk` 500000x100 1.98x;
Ridge predict `dgemv` 20Mx20 1.77x; newton-cholesky `dgemm` 2199x40000x2199
1.67x. OpenBLAS wins only on small f32 `sgemv` predicts (~2x). KMeans even.

## Thread-count patterns

- At equal pinned cores on the Xeon, Ridge fit and KMeans are within ~1.2x
  up to 16 cores: the single-node kernels are close.
- Past one NUMA node (>= 32 cores), OpenBLAS `dsyrk` stops scaling (~2x
  behind MKL) while MKL's thin `dgemm` anti-scales (2-3x behind OpenBLAS).
- The OpenMP OpenBLAS build loses more than the pthreads build on small and
  medium linear-model ops.
- Geomean OpenBLAS/MKL on fit at default threads: LinearRegression
  1.4-1.9x, Ridge 1.3-1.6x, LogisticRegression 1.3-1.7x, KMeans 0.9-1.16x.
