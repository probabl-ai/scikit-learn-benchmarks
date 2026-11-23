# Audit of the Phoronix Test Suite scikit-learn profile

The current PTS profile (`pts/scikit-learn-2.0.0`, May 2023) looks broken on recent hardware more than skewed by code outside scikit-learn. On one socket of our Xeon 6787P, 14 of its 40 entries gave no score: 9 hit a 40 minute cap per run and 5 fail every run.

- The install silently degrades: the profile's OpenBLAS doesn't build on this Xeon nor on a Core Ultra 9 285, and PTS still reports success, with numpy built without BLAS. 6 of the 9 timeouts seem to come from that.
- Time outside scikit-learn seems small for most entries: 18 of the 26 scored entries spend over 94% of their time inside scikit-learn calls. A few are dominated by data generation or loading (Plot SVD, Isolation Forest), but the missing BLAS inflates the data generation part.
- LocalOutlierFactor and probably TSNE can't work with more than 64 threads (scipy's OpenBLAS wheel crashes), and Glmnet, Plot NMF and RCV1 Logreg seem broken on any machine.

Run by an agent: real PTS client and unmodified profile, Python 3.11, pip pinned to the profile's date, one socket of intel-xeon (172 threads), py-spy on every run, one sweep.

Open question: what to put in the Phoronix email. If we need the outside-scikit-learn shares as they were on 2023 hardware, the next step is a second sweep with a working BLAS (the numpy 1.24.3 PyPI wheel).

## Details

### Question

We want to propose a subset of `sklbench` to Phoronix for the scikit-learn part of PTS. The question was how much of each PTS score is spent outside `fit`/`predict` (imports, data loading and generation, plotting, script overhead), and in general how the current profile behaves on current hardware.

### How the profile works

- `pts/scikit-learn-2.0.0` (10 May 2023) has 40 menu entries, each running one `bench_*.py` script of scikit-learn 1.2.2's `benchmarks/` folder, with a 1,900-line patch (`scikit-learn-benchmark-improvements-1.tar.xz`).
- Scoring: the profile has a `<SystemMonitor><Sensor>sys.time</Sensor>` monitor. When monitoring produces a result, PTS skips `parse_result` (`pts_test_execution.php`, around L327 and L392). So the score is the wall time of the whole `python3 bench_*.py` process, and the profile's LineHint parsers are dead code.
- Runs: `TimesToRun=3`, `IgnoreRuns=1` (the first run is discarded), plus PTS's dynamic run count. With the default config (`StandardDeviationThreshold=2.5`, `LimitDynamicToTestLength=20` minutes), entries whose runs vary by more than 2.5% get up to 4x, 3x or 2x the planned runs depending on their length, and then up to 3 more (20 Newsgroups ran 16 times, Plot SVD and Plot Lasso Path 10 times).
- Install chain:
  1. Builds OpenBLAS commit 437c0bf from source (`NUM_THREADS=512`, OpenMP, `DYNAMIC_ARCH=0`).
  2. Builds numpy 1.24.3 against it.
  3. Tries to build a scipy snapshot (20230505), joblib 1.2.0, threadpoolctl 3.1.0 and scikit-learn 1.2.2 with `pip3 install --user`.
- OpenBenchmarking.org has data for 24 of the 40 entries, all dated 10 May to 25 Dec 2023. The 16 others have no public result.

### What breaks in the install (confirmed)

- numpy 1.24.3 pins `setuptools==59.2.0`, which can't be imported on Python 3.12 or later (no `distutils`): the profile can't install on Python 3.12+ at all (Ubuntu 26.04 has 3.14).
- OpenBLAS 437c0bf fails to build:
  - Core Ultra 9 285 (Arrow Lake): fails in `getarch_2nd`.
  - Xeon 6787P: detected as `PRESCOTT`, then `sgemm_oncopy.o` fails with `inlining failed in call to 'always_inline' '_mm512_add_ps': target specific option mismatch` (conda-forge GCC 12.4; system GCC 16 fails too, with incompatible pointer errors).
  - PTS doesn't stop: numpy is then built with no BLAS at all (`numpy.show_config()` shows every BLAS/LAPACK as `NOT AVAILABLE`, so numpy uses its unoptimized fallback dot and `lapack_lite`).
- The `mesonpy` package in the pip line doesn't exist on PyPI, so that whole pip command fails.
- The scipy snapshot build fails (`Include dir ../../f2py/src does not exist`, because `$NUMPYDIR` is empty). pip then installs the scipy 1.10.1 wheel as a scikit-learn dependency. Its bundled OpenBLAS 0.3.18 runs its `Prescott` kernels on both CPUs (threadpoolctl: `architecture: Prescott`), and it is built for at most 64 threads.
- The `glmnet` pip package doesn't build, so the Glmnet entry fails with `No module named 'glmnet'`.

Final stack on xeon (`results/logs/xeon-s1/versions.txt`): Python 3.11.18, numpy 1.24.3 (no BLAS), scipy 1.10.1 (OpenBLAS 0.3.18, Prescott, 64 threads), scikit-learn 1.2.2, joblib 1.2.0, threadpoolctl 3.1.0, libgomp with 172 threads.

### Setup

- Machine: intel-xeon (Xeon 6787P, Ubuntu 26.04), benchlock slot `s1` (one socket, 172 logical CPUs, NUMA nodes 4-7). Socket-slot results, not for the dashboards.
- Harness in `harness/`:
  - `pixi.toml`: Python 3.11, pip >= 26.1, PHP, cmake, meson, ninja, conda-forge GCC/G++/GFortran 12, py-spy.
  - `run_all.sh`: runs the real PTS client from a git clone (`php -d disable_functions=posix_getpwuid`, else PTS ignores `$HOME`) with a private `HOME`, so `pip --user` and `scikit_learn_data` stay in the work dir. `PIP_UPLOADED_PRIOR_TO=2023-05-10T00:00:00Z` makes pip resolve as on the profile's date. `RunAllTestCombinations=FALSE` is needed for `PRESET_OPTIONS_VALUES` to apply in batch mode. It runs `batch-install`, then one `batch-run` per entry. `I0` offsets the entry numbers when resuming.
  - `shim/python3`: first on `PATH`. For `bench_*.py` calls, it starts Python, attaches `py-spy record --pid --idle --threads --nonblocking --rate 50 --format raw`, kills the run after `RUN_LIMIT_S` (2,400 s) and skips the later runs of an entry that hit the cap (exit 124), and appends one JSON line per run to `runs.jsonl`. Other calls go to the real Python.
  - `sitecustom/sitecustomize.py`: makes `os.cpu_count()` return the affinity mask size (172 in the slot), and calls `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` so py-spy can attach under `ptrace_scope=1`.
  - `analyze.py`: splits main-thread samples of the counted runs (run >= 1, rc 0) into categories, first match wins: `import` (importlib frames), `sklearn.datasets`, `sklearn`, `matplotlib`, `other`. `unsampled` is `wall_s * 50` minus the samples. It reads the PTS scores from the result files.
  - `top_timedout.py`, `lof_threads.sh`, `check_caches.py`: the checks below.
- Differences from a stock PTS run: Python 3.11 and GCC 12 from conda-forge (the system Python can't install the profile), pip resolution pinned to 2023-05-10, private home, one socket with `os.cpu_count()` matching it, 40 minute cap per run, py-spy attached (it samples from another process; its overhead was not measured).

Commands (on intel-xeon, in `~/work/pts-sklearn-audit`, a copy of `harness/` plus a PTS clone in `ptsrc/`):

```bash
RUN_ID=xeon-s1 RUN_LIMIT_S=2400 nohup benchlock --tag "PTS scikit-learn audit" s1 -- \
    bash -lc "pixi install && pixi run ./run_all.sh" > run.log 2>&1 < /dev/null &
# resume with the remaining entries:
RUN_ID=xeon-s1 I0=40 ... pixi run ./run_all.sh entries_rest.txt
pixi run python analyze.py . xeon-s1   # writes analysis-xeon-s1.{md,json}
```

### Results

The sweep ran from 17 to 23 Nov 2026 (stopped twice to free the machine): 189 `bench_*.py` runs, 25.0 h of script wall time. Score = PTS average of the counted runs. Shares are of main-thread py-spy samples over the counted runs.

| Entry | Score (s) | OpenBenchmarking median (s) | in sklearn | Main cost outside sklearn |
|---|---:|---:|---:|---|
| 20 Newsgroups / Logistic Regression | 49.2 | 42.1 | 96% | loading the cached vectorized data, 2% |
| Covertype Dataset Benchmark | 315.6 | 381 | 99% | |
| Feature Expansions | 186.5 | 142 | 86% | scipy `sparse.random` + `tocsr` (input data), 13% |
| GLM | cap | 373 | | |
| Hist Gradient Boosting | cap | 98.25 | | |
| Hist Gradient Boosting Adult | cap | 84.07 | | |
| Hist Gradient Boosting Categorical Only | cap | 17.61 | | |
| Hist Gradient Boosting Threading | 167.3 | 174 | 94% | `make_classification`, 6% |
| Lasso | cap | 430 | | |
| Plot Hierarchical | 207.3 | 198 | 99% | |
| Plot Incremental PCA | 302.2 | 47.94 | 98% | |
| Plot Lasso Path | 1146.5 | 225 | 29% | `make_regression` 40%, numpy called by the script 31% |
| Plot Neighbors | 285.8 | 141 | 99% | |
| Plot OMP vs. LARS | 1143.4 | 87.32 | 55% | the script's Gram matrix `np.dot(X.T, X)` (inside its timed region), 44% |
| Plot Ward | 65.1 | 58.57 | 49% | scipy `ward` (compared on purpose), 49% |
| Sparse Random Projections / 100 Iterations | cap | 811 | | |
| SAGA | 687.2 | 810 | 98% | |
| Sample Without Replacement | 114.2 | 120 | 0% | 99% "own code", see caveats |
| SGD Regression | cap | 102 | | |
| Sparsify | 113.1 | 101 | 98% | |
| Text Vectorizers | 55.5 | 58.56 | 95% | |
| Tree | 44.9 | 42.39 | 96% | |
| MNIST Dataset | 59.8 | 59.81 | 98% | |
| TSNE MNIST Dataset | fails | 281 | | |
| Glmnet | fails | | | |
| Hist Gradient Boosting Higgs Boson | 670.0 | | 52% | 47% unsampled, see caveats |
| Isolation Forest | 282.2 | | 38% | loading kddcup99 from the local cache, 51%; script, 7% |
| Isotonic / Perturbed Logarithm | 1918.7 | | 98% | |
| Isotonic / Logistic | 1708.9 | | 90% | numpy called by the script, 7% |
| Isotonic / Pathological | cap | | | |
| Kernel PCA Solvers / Time vs. N Components | 459.9 | | 99% | |
| Kernel PCA Solvers / Time vs. N Samples | 1023.2 | | 100% | |
| LocalOutlierFactor | fails | | | |
| SGDOneClassSVM | 274.3 | | 95% | `fetch_kddcup99`, 5% |
| Plot Fast KMeans | 724.6 | | 96% | script, 3% |
| Plot Non-Negative Matrix Factorization | fails | | | |
| Plot Parallel Pairwise | cap | | | |
| Plot Polynomial Kernel Approximation | 364.4 | | 99% | |
| Plot Singular Value Decomposition | 652.1 | | 6% | `make_low_rank_matrix` (input data), 90% |
| RCV1 Logreg Convergence | fails | | | |

Full table with every category: `results/analysis-xeon-s1.md`; per-entry data (per-run PTS values, top callees): `results/analysis-xeon-s1.json`; every run: `results/prof/xeon-s1/runs.jsonl`.

### Where the capped runs spent their time

From `top_timedout.py` (main-thread samples of the first run, cut at 2,400 s):

| Entry | Hot spot |
|---|---|
| GLM | 91% `safe_sparse_dot` in `Ridge.fit`, 7% scipy `lstsq` |
| Lasso | 98% numpy (BLAS-less dot) inside `Lasso.fit` and `LassoLars.fit` |
| Sparse Random Projections | 100% `GaussianRandomProjection.transform` (dot) |
| SGD Regression | 96% `safe_sparse_dot` (script line 122) |
| Plot Parallel Pairwise | 100% `safe_sparse_dot` in `euclidean_distances` |
| Isotonic / Pathological | 77% `isotonic_regression`, 11% `_asarray_with_order` |
| HGB, HGB Adult, HGB Categorical Only | 97-100% in `TreeGrower.split_next` and `_compute_best_split_and_push` |

- The first 5 are numpy matrix products without BLAS.
- The 3 HGB entries don't use BLAS. They are at least 24-136x slower than the OpenBenchmarking medians, while HGB Threading (also 172 threads) is at the median. My guess is OpenMP overhead with 172 threads on small trees, not checked. py-spy only recorded 46-51% of the expected samples on these runs.

### Failures

| Entry | Error | Cause |
|---|---|---|
| LocalOutlierFactor | segfault (rc 139) after ~7 s | OpenBLAS 0.3.18 (scipy wheel, built for 64 threads) called from 172 OpenMP threads; confirmed below |
| TSNE MNIST Dataset | segfault after ~220 s | very likely the same: its profile has ~40 s in `KNeighbors` (`ArgKmin`) before the crash; not tested at fewer threads |
| Glmnet | `ModuleNotFoundError: No module named 'glmnet'` | the `glmnet` package doesn't build |
| Plot NMF | `KeyError: <Axes: >` after ~77 s | matplotlib/pandas error at plotting time, with the 2023-05-10 package versions |
| RCV1 Logreg Convergence | `IndexError: list index out of range` after ~12 s | in the script, not investigated |

LOF thread test (`harness/lof_threads.sh`, slot `s0`, same install):

| `OMP_NUM_THREADS` | Result |
|---|---|
| unset (172) | segfault after 8 s, `OpenBLAS warning: precompiled NUM_THREADS exceeded` repeated |
| 64 | OK, 32 s |
| 32 | OK, 35 s |

Logs in `results/lof_*.log`. This is the same OpenBLAS buffer exhaustion as scikit-learn's KMeans on many-core machines (OpenMathLib/OpenBLAS#5958, see `configs/_real_datasets.py`).

### Problems with the harness, and how they were handled

- I copied `~/scikit_learn_data` into the private home to save downloads. The 20newsgroups vectorized and RCV1 pickles were written by numpy 2 (`No module named 'numpy._core'`), and the kddcup99 cache was invalid for scikit-learn 1.2.2. 20 Newsgroups and Isolation Forest failed on the first pass. I moved those caches to `stale-numpy2-caches/`, moved the failed runs to `prof/xeon-s1/failed-numpy2-cache/` and `failed-first-pass/`, and reran both entries at the end. The other entries using these datasets (SAGA, SGDOneClassSVM, LOF, RCV1) ran after the fix and downloaded their own data.
- Higgs Boson: the script downloads the 2.8 GB `HIGGS.csv.gz` from UCI with `urlretrieve` during its first run, with no size or checksum check. The connection dropped at 806 MB and the next runs failed on the truncated file (`EOFError: Compressed file ended before the end-of-stream marker was reached`). I downloaded it again (2,816,407,858 bytes, `gzip -t` OK) and reran the entry. A PTS user with a flaky connection would hit the same failure.
- The sweep was stopped twice to free intel-xeon (SGDOneClassSVM and Higgs Boson were cut mid-entry, their partial runs are in `prof/xeon-s1/failed-first-pass/` and `interrupted/`, and both were rerun from scratch).
- On intel-gamer-gpu, an early trial left an orphaned PTS process (PTS renames itself to "Phoronix Test Suite", so `pkill -f phoronix-test-suite.php` missed it). It ran the 40-entry sweep outside benchlock from ~17:08 to 21:43 UTC on 17 Nov 2026. No GitHub workflow used that machine in that window; ad hoc runs by other agents can't be ruled out.

### Caveats

- One sweep, one machine, one socket. Scores and shares depend on the broken BLAS: data generation and numpy-in-script shares (Plot SVD, Plot Lasso Path, Plot OMP vs. LARS) would be smaller with a working BLAS. The Isolation Forest loading share doesn't use BLAS.
- py-spy without `--native` doesn't see compiled frames: time in a Cython function called from the script is attributed to the script line. Sample Without Replacement's "99% own code" is probably scikit-learn's Cython `sample_without_replacement` plus the stdlib and numpy alternatives the script compares, not overhead.
- Higgs Boson: py-spy recorded only 50-56% of the expected samples (17-19k of 33-35k per run, 3-8 errors), cause unknown. The recorded samples are almost all in scikit-learn. The parsed CSV is cached with joblib `Memory` by the discarded first run, so the counted runs only load the cache.
- The OpenBenchmarking medians come from 2023 hardware with (presumably) a working BLAS, and from whole machines, so they are only a rough reference.
- Shares are of main-thread samples; worker threads are covered by the main thread's call.

### Raw data

On intel-xeon in `~/work/pts-sklearn-audit/`: `prof/xeon-s1/` (py-spy raw profiles, 30 MB), `logs/xeon-s1/` (PTS logs per entry, install logs), `home/.phoronix-test-suite/test-results/skl-xeon-s1-*` (PTS result files), `run.log`, `run2.log`, `run3.log`. This folder has the summary files only.

### Possible next steps

- Second sweep with the numpy 1.24.3 PyPI wheel (bundled OpenBLAS) in place of the BLAS-less build, on slot `s0`, ~10 h: gives the shares as on 2023 hardware.
- Check the HGB slowdown at 172 threads (rerun HGB Adult at 172, 64 and 16 threads).
- Run TSNE at 64 threads to confirm it is the same crash as LOF.
