## 2026-09-30 progress meeting

Follow-up:

- [https://github.com/uxlfoundation/oneDAL/pull/3773](https://github.com/uxlfoundation/oneDAL/pull/3773): david to take over this PR? => Anatoly took it over
- In joblib: [_prepare_worker_env(n_jobs)](https://github.com/joblib/joblib/blob/683c83ed54e8eea2b86af1a5f249c827abbb9426/joblib/_parallel_backends.py#L221) does NOT limit the number of threads if inner parallelism is joblib threads (nor joblib loky). But Itamar is adding that
- Torch-CPU LogisticRegression slower: overhead of parallelism on every op (not just mat-mul like numpy). With parallelism disabled, it closely matches numpy’s speed (=> array API doesn’t have a big overhead in itself). I would be curious to see what torch.compile can do.

Performance work:

- FYI: [https://github.com/scipy/scipy/pull/26193#issuecomment-5886021070](https://github.com/scipy/scipy/pull/26193#issuecomment-5886021070) LogisticRegression in PyPI wheel suffers from parallelism overhead that was fixed in more recent OpenBLAS, scipy was still bundling an old OpenBLAS, they bumped recently
- OneDal logistic reg bug: [https://github.com/uxlfoundation/oneDAL/issues/3820](https://github.com/uxlfoundation/oneDAL/issues/3820)
- Isnan fix for conda-forge merged: [https://github.com/scikit-learn/scikit-learn/pull/34876](https://github.com/scikit-learn/scikit-learn/pull/34876)
- Started experimenting with radix sort in trees: very promising [results on the laptop](https://pr-126-compare-intel-laptop.sklbench-pr-comparison.pages.dev/pr_comparison%20), but less good [on the GNR](https://pr-126-compare-intel-gnr.sklbench-pr-comparison.pages.dev/pr_comparison) for now.
- Polars experiment on the GNR: [https://github.com/joblib/joblib/issues/1548#issuecomment-5910160662](https://github.com/joblib/joblib/issues/1548#issuecomment-5910160662)
  - Bad scalability without nesting polars under Python thread
  - Nesting parallelism is as bad as without nesting.
- HGB thread management improvement seems to make scikit-learn close to Pareto optimal vs xgboost/lightgbm/catboost: [https://github.com/scikit-learn/scikit-learn/pull/34935#issuecomment-5817871757](https://github.com/scikit-learn/scikit-learn/pull/34935#issuecomment-5817871757)

Dashboard: making it ready for communication

- [https://probabl-ai.github.io/scikit-learn-benchmarks/](https://probabl-ai.github.io/scikit-learn-benchmarks/)
- WIP Polishing visuals/functionalities/names of machines/software environment etc.
- Adding explanatory texts, findings, links to issues, etc.

Joblib / loky / threadpoolctl & free threading

- WIP to reduce threadpoolctl overhead
- Released loky 3.7.0 with a threadlocal reusable process pool to allow:
- Thread-safe joblib [https://github.com/joblib/joblib/pull/1834](https://github.com/joblib/joblib/pull/1834)
- WIP review to mitigate oversubscription between nested joblib managed process pools and thread pools:
  - [https://github.com/joblib/joblib/pull/1825](https://github.com/joblib/joblib/pull/1825)

TODO:

- Open “RFC Active wait capabilities” issue in threadpoolctl (ping David/Vic/Anatoly there).
  - Q: Global settings or thread-local?
- PyTorch statically linked against MKL cannot be controlled by threadpoolctl based oversubscription protection in joblib.
- Consider benchmarking sparse penalties for logistic regression
- Consider benchmarking logistic regression with sparse data (large number of features)
- Highway-based dynamic dispatch for radix sort?

## 2026-09-23 progress meeting

Arthur was off most of the week (from Friday to Tuesday included)

Performance work:

- Iterations in Ridge optimization about numerical stability: [https://github.com/scikit-learn/scikit-learn/pull/34793](https://github.com/scikit-learn/scikit-learn/pull/34793)
- HGB scalability fixes still under review (waiting for Olivier’s feedback):
  - [https://github.com/scikit-learn/scikit-learn/pull/34935](https://github.com/scikit-learn/scikit-learn/pull/34935)
- Inlined isnan: [https://github.com/scikit-learn/scikit-learn/pull/34876](https://github.com/scikit-learn/scikit-learn/pull/34876) (WIP)

Free-threading / threadpoolctl / loky / joblib

- Follow-up reviews to fix performance overhead issues of threadpoolctl after the 3.7.0 release.
- WIP: reviewing thread-safety fixes in loky to make joblib fully thread-safe, whatever the backend and enabled thread-safety stress testing on its CI
- Loky cpu_count improved for physical cores detection (might be particularly useful for big machines on linux). Useful for our benchmarking setup but also for joblib end users on big data center grad machines with complex deployment (cgroups, taskset…)
- Next steps: review work to integrated threadpoolctl based oversubscription mitigation by default in joblib.

Benchmarking infrastructure

- Previous results disappeared from the dashboard: should be fixed now.
- New benchmark to study scalability of HP tuning (grid search): [https://probabl-ai.github.io/scikit-learn-benchmarks/](https://probabl-ai.github.io/scikit-learn-benchmarks/)
- FYI: will soon collect runtimes for array API on CUDA device and include that to the comparison.
- FYI: shall reuse this repo to also benchmark TabICL?
- TODO: add xgboost (and maybe ligtgbm, catboost) to the benchmark mix :)
  - If we do, we should also benchmark CUDA vs Intel GPU.

TODO: contact xgbboost about conda-forge flags

## 2026-09-16 progress meeting

Performance work:

- Conda-forge build slower for ExtraTrees fix: [https://github.com/scikit-learn/scikit-learn/pull/34876](https://github.com/scikit-learn/scikit-learn/pull/34876)
- HGB scalability: [https://github.com/scikit-learn/scikit-learn/pull/34935](https://github.com/scikit-learn/scikit-learn/pull/34935)
- Ridge optimization might be not that easy (numerical stability) =>
  - Check skrub table-vectorizer
  - Check if we have the same problem elsewhere (PCA? …) with similar centering-trick

- Threadpoolctl 3.7.0 released:
  - Configure MKL to set thread-local limits by default to make it possible to avoid oversubscription under free-threading Python threads.
  - Similarly for OpenBLAS with OpenMP
  - Other fixes for deadlocks and crashes on linux and windows
  - [https://github.com/joblib/threadpoolctl/blob/master/CHANGES.md#370-2026-09-15](https://github.com/joblib/threadpoolctl/blob/master/CHANGES.md#370-2026-09-15)
  - [https://bsky.app/profile/ogrisel.bsky.social/post/3mvmrfr54kc2m](https://bsky.app/profile/ogrisel.bsky.social/post/3mvmrfr54kc2m)
- Benchmark collection:
  - Measured HGB scalability on Apple M4 to check that [#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) works as expected on that machine.
  - Measuring array API on Apple M4 at the moment.

Follow-ups:

- OpenBLAS + GNU OpenMP? Possible to build but not possible to get through conda-forge
- Is [this](https://github.com/scikit-learn/scikit-learn/blob/8a6f766c6dbfa2b443ac1db30581b7129a1fc456/sklearn/cluster/_kmeans.py#L906) (_warn_mkl_vcomp) still current? => Probably yes
- treelite + decision trees with missing values, the bug is on the treelite side [https://github.com/dmlc/treelite/issues/706](https://github.com/dmlc/treelite/issues/706) (and fixed already)

For reference, items on Sergey’s side:

## 2026-09-02 progress meeting

Array API:

- Work in progress on numerical stability confusion_matrix_at_thresholds:
  - [https://github.com/scikit-learn/scikit-learn/issues/34813](https://github.com/scikit-learn/scikit-learn/issues/34813)
  - 1 PR merged (unweighted case), 1 under review/discussion (weighted case)
- Revived old integration test PR
  - [https://github.com/scikit-learn/scikit-learn/pull/32873](https://github.com/scikit-learn/scikit-learn/pull/32873)
- Continued discussion on fitted attributes: pending performance evaluation by Tim.

Performance work/Benchmarks follow-up:

- Issue on oneDal related to best-first splitting strategy in RFs: [https://github.com/uxlfoundation/oneDAL/issues/3771](https://github.com/uxlfoundation/oneDAL/issues/3771)
  - Fixes: [#3772](https://github.com/uxlfoundation/oneDAL/pull/3772) (in review); [#3773](https://github.com/uxlfoundation/oneDAL/pull/3773) (draft)
- Scikit-learn intelex issues:
  - [https://github.com/uxlfoundation/scikit-learn-intelex/issues/3402](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3402)
  - [https://github.com/uxlfoundation/scikit-learn-intelex/issues/3401](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3401)
- Conda-forge build slower for ExtraTrees: investigation finished:
  - [Investigation: conda-forge scikit-learn slower than PyPI (ExtraTrees case)](https://github.com/scikit-learn/scikit-learn/issues/34869)
  - [Investigation: Meson silently downgrades -O3 to -O2 under conda-forge CFLAGS](https://github.com/scikit-learn/scikit-learn/issues/34865)
  - Follow-up? Benchmark compilation flags

Benchmarks:

- Models scalability: [https://probabl-ai.github.io/scikit-learn-benchmarks/models_scalability.html](https://probabl-ai.github.io/scikit-learn-benchmarks/models_scalability.html)
  - Feedback from Olivier: it would help to share the y axis (fit durations) for different plots of the same model on different software.
- Full re-run with latest configs fixes, etc.

TODO:

- Investigate treelite + decision trees with missing values
- Investigate logistic regression time break-down: maybe lots of time is spent in openblas thread lock mechanisms (checkable through VTune).

Is this still current? [https://github.com/scikit-learn/scikit-learn/blob/8a6f766c6dbfa2b443ac1db30581b7129a1fc456/sklearn/cluster/_kmeans.py#L906](https://github.com/scikit-learn/scikit-learn/blob/8a6f766c6dbfa2b443ac1db30581b7129a1fc456/sklearn/cluster/_kmeans.py#L906)

## 2026-08-26progress meeting

Optimization work in scikit-learn:

- Ridge: [https://github.com/scikit-learn/scikit-learn/pull/34793](https://github.com/scikit-learn/scikit-learn/pull/34793)
- HGB (still WIP but almost ready):
  - PR: [https://github.com/scikit-learn/scikit-learn/pull/34788](https://github.com/scikit-learn/scikit-learn/pull/34788)
  - Added calibration to “detect” e.g. lack of active waits

Benchmarks:

- Reported issue on Ridge CPU: [https://github.com/uxlfoundation/scikit-learn-intelex/issues/3377](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3377)
- Loky improvement (physicial cores detection under taskset): [https://github.com/joblib/loky/pull/651](https://github.com/joblib/loky/pull/651)

Array API:

- Discussion about containers for fitted attributes (move to numpy by default):
  - [https://github.com/scikit-learn/scikit-learn/issues/34604](https://github.com/scikit-learn/scikit-learn/issues/34604)
- .score and pandas Series merged
- Common test for mixed namespace estimators pass for all estimators
- Array API support ROC AUC score in progress
- Work in progress on numerical stability confusion_matrix_at_thresholds.

Possible future step: investigate optimization flags in conda-forge that impact decision trees.

## 2026-08-19progress meeting

OMP & HGB scalability investigations:

- State of my current knowledge [https://github.com/scikit-learn/scikit-learn/issues/34764](https://github.com/scikit-learn/scikit-learn/issues/34764)
- WIP: mitigation strategies (adequate threadpool sizing & serial mode for small workloads)
  - Discussed: static linking of OpenBLAS with OpenMP for KMeans?
  - Other possibility: change the manylinux image to give an option to build with llvm openmp to benefit from runtime change of the active wait setting (the KMP blocktime)?
- If we cannot fully fix the problem we could improve performance guideline documentation.

Benchmarks:

- WIP HGB investigation & measurement of gains from mitigation strategies
- Discussed: compare to lightgbm with and without active waiting enabled.

Array API:

- Fixing pandas labels bug in .score:
  - [https://github.com/scikit-learn/scikit-learn/pull/34779](https://github.com/scikit-learn/scikit-learn/pull/34779) (waiting for review)
- Array API support for ROC AUC score under review:
  - [https://github.com/scikit-learn/scikit-learn/pull/33200](https://github.com/scikit-learn/scikit-learn/pull/33200)
  - Need to check if there are good numerical stability test.
- Other reviews related to fix the mixed namespace inputs support:
  - [https://github.com/scikit-learn/scikit-learn/issues/28668](https://github.com/scikit-learn/scikit-learn/issues/28668)
  - Most PR merged, still one small to review.
- For information: we are discussing the default namespace device for fitted attribute.
  - [https://github.com/scikit-learn/scikit-learn/issues/34604](https://github.com/scikit-learn/scikit-learn/issues/34604)

TabICL updates:

- Device handling refactoring still waiting for review.
- PoC on memory usage improvements for the CPU:

ARM paper on scikit-learn acceleration through oneDAL: [https://arxiv.org/pdf/2504.04241](https://arxiv.org/pdf/2504.04241)

## 2026-08-12progress meeting

### TabICL preliminary work

- Refactored device handling to remove cuda-specific assumptions
  - PR: [https://github.com/soda-inria/tabicl/pull/144/](https://github.com/soda-inria/tabicl/pull/144/) (ready for review)
  - Evaluation results: [https://gist.github.com/ogrisel/5de077483055ad451ac7cce47b3356d6](https://gist.github.com/ogrisel/5de077483055ad451ac7cce47b3356d6)
- Good results for laptop GPU via the torch xpu device:
  - AMP => half the memory usage than CPU
  - Intel laptop GPU (Arc B390) with xpu 3x faster than laptop CPU (Core Ultra x7 358H)
  - Intel laptop GPU with xpu 2x faster than Apple M4 GPU via the torch mps device
- Possible TODOs for later:
  - Dedicated Intel GPU CI
  - Profile and optimize further both for CPU and GPU devices.
  - More memory efficiency optimization.
  - Try OpenVINO for inference with quantization?

**Benchmarks:**

- collected benchmark results on GNR
- WIP: alternative to py-spy for profiling (cProfile, samply?)
- Builds comparison: [https://probabl-ai.github.io/scikit-learn-benchmarks/builds_comparison.html](https://probabl-ai.github.io/scikit-learn-benchmarks/builds_comparison.html)
  - => on the GNR, mkl brings a clear speed-up for linear models & kmeans
- HGB scaling plots & analysis:
  - [https://probabl-ai.github.io/scikit-learn-benchmarks/hgb_build_scaling.html](https://probabl-ai.github.io/scikit-learn-benchmarks/hgb_build_scaling.html)
    - Reveals active wait policy impact
    - But even with active wait, scaling is bad past a certain number of threads
  - [https://probabl-ai.github.io/scikit-learn-benchmarks/hgb_scaling.html](https://probabl-ai.github.io/scikit-learn-benchmarks/hgb_scaling.html)

**Encoders performance improvements:**

- Preliminary refactor: [https://github.com/scikit-learn/scikit-learn/pull/34452](https://github.com/scikit-learn/scikit-learn/pull/34452) (merged)
- Pandas fast path: [https://github.com/scikit-learn/scikit-learn/pull/34678](https://github.com/scikit-learn/scikit-learn/pull/34678) (ready for review)
  - Alternative with narwhals: [https://github.com/scikit-learn/scikit-learn/pull/34721](https://github.com/scikit-learn/scikit-learn/pull/34721) => slowdown for small/medium datasets, but includes polars/pyarrow

**(Very) Exploratory performance work (WIP/planned):**

- HGB: mitigate effect of “no-active wait builds”, improve overall scalability
- SIMD sort for trees: [https://github.com/scikit-learn/scikit-learn/pull/34693](https://github.com/scikit-learn/scikit-learn/pull/34693)
  - Dynamic dispatch enabled or not in those experiments?
  - Instead of calling it from Cython we could call it from Python (via numpy) to avoid introducing extra packaging maintenance / complexity. But we need a kv-sort that might not be exposed by the Python API in NumPy.
- Alternative: consider radix sorts instead (after mapping values to their ranks)

## 2026-08-05progress meeting

**Benchmarks:**

- Added [real datasets](https://github.com/probabl-ai/scikit-learn-benchmarks/blob/main/configs/real_datasets.py) (~10) with more [realistic pre-processing](https://github.com/probabl-ai/scikit-learn-benchmarks/blob/main/sklbench/runners/datasets/preprocessing.py).
  - Full pipeline benchmarking will come after with a dedicated section in the result dashboard. Polars could help for faster preprocessing.
  - Improved polars support via narwahls will be needed to benefit from using polars:
    - [https://github.com/cakedev0/scikit-learn/pull/9](https://github.com/cakedev0/scikit-learn/pull/9) (still under review)
- Improved synthetic cases matrix to cover a higher variety of cases while preserving a decent runtime.
- Thanks to the 2 points above, found various bugs:
  - report issue: nan in X only at predict time in scikit-learn-intelex crashes: [https://github.com/uxlfoundation/scikit-learn-intelex/issues/3356](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3356)
  - New issue reported to PyTorch XPU: [https://github.com/intel/torch-xpu-ops/issues/4805](https://github.com/intel/torch-xpu-ops/issues/4805)
  - Many other discovered bugs in pytorch XPU 2.12/13 were fixed in `main` (future 2.14)
    - Advise dpnp in sklearn doc? Or at least latest pytorch [https://github.com/scikit-learn/scikit-learn/issues/34648](https://github.com/scikit-learn/scikit-learn/issues/34648)
- WIP: collecting benchmark results on GNR
  - Timeouts with scikit-learn RFs to investigate
  - KMeans with OpenBLAS crash
    - [https://github.com/OpenMathLib/OpenBLAS/issues/5958](https://github.com/OpenMathLib/OpenBLAS/issues/5958)
    - Possible workaround with taskset limits for the KMeans runs.
- CI with manual dispatch to run benchmarks (only on the laptop for now) => how should we configure the runner on the GNR?

**Array API**

- Reviewing PRs to update existing classifiers to pass the new common tests:
  - [https://github.com/scikit-learn/scikit-learn/pull/34480](https://github.com/scikit-learn/scikit-learn/pull/34480) (merged)
  - [https://github.com/scikit-learn/scikit-learn/pull/34475](https://github.com/scikit-learn/scikit-learn/pull/34475) (WIP)

**Multithreading in scikit-learn**

- Maintenance work required in loky to fix thread-safety problems.
- TODO: document thread-safety expectations for the scikit-learn code base. Fit is expected to not be thread safe but predict and transform should be stateless and therefore thread safe. There might be exception in case of silent caching at prediction time. We would need testing and documentation.

## 2026-07-29progress meeting

**Encoders performance improvements:**
Pandas/Series fast paths improvements were harder to make work than expected:

- Preliminary refactor: [https://github.com/scikit-learn/scikit-learn/pull/34452](https://github.com/scikit-learn/scikit-learn/pull/34452) (in review)
- [https://github.com/scikit-learn/scikit-learn/pull/34393](https://github.com/scikit-learn/scikit-learn/pull/34393) closed in favor of a more generic approach based on narwhals: [https://github.com/cakedev0/scikit-learn/pull/9](https://github.com/cakedev0/scikit-learn/pull/9) (ready but waiting for preliminary refactor to be merged)
- Small batches transform speed-up: [https://github.com/cakedev0/scikit-learn/pull/13/](https://github.com/cakedev0/scikit-learn/pull/13/) (WIP)

**Array API**

- Follow-up from last update:
- testing infra for mixed input namespaces: [https://github.com/scikit-learn/scikit-learn/pull/34439](https://github.com/scikit-learn/scikit-learn/pull/34439) merged
- array API support in the newton-cg solver of LogisticRegression: [https://github.com/scikit-learn/scikit-learn/pull/34412](https://github.com/scikit-learn/scikit-learn/pull/34412) merged
- Related work from other contributors:
- Better test coverage of Array API metrics: [https://github.com/scikit-learn/scikit-learn/pull/34442](https://github.com/scikit-learn/scikit-learn/pull/34442) merged
- Implement "everything follows X" for GridSearchCV: [https://github.com/scikit-learn/scikit-learn/pull/33633](https://github.com/scikit-learn/scikit-learn/pull/33633) merged

**Benchmarks:**

- In progress: adding cases with real datasets and meaningful/realistic preprocessing

**Questions:**

- C++ pieces of scikit-learn intelex in scikit-learn?
- Comment from [Cortes, David](mailto:david.cortes@intel.com): Might be quite a challenge (e.g. dynamic dispatching achieved through multiple compiles of the same file, CPU features detection, aligned memory allocators from TBB, etc.). If used as a base, some of these things could be done differently (e.g. usage of ‘target_clones’ instead of multiple compiles, usage of OMP instead of TBB, usage of exceptions instead of error codes, etc.) in a way that’s more compatible with standard compilers and build systems. Could provide some help if needed with identifying and addressing these cases, but would not be too easy. Note that there might be large refactoring of forests from Intel’s team in the future, particularly related to parallelization.

**Points from David:**

- What is supposed or not to be thread-safe in scikit-learn? Are we planning to write guidelines/documentation about that? To [Olivier Grisel](mailto:olivier@probabl.ai)
- [https://www.openmp.org/2026/python-subcomittee/](https://www.openmp.org/2026/python-subcomittee/): should we get involved?
- Intel GPUs should be competitive for float64 (better support than nvidia?)
- GNR is equipped with the cheapest desktop GPU on the market; we could maybe hand down benchmark repo & instructions for David to run on a server GPU at some point.

## 2026-07-08progress meeting

**Investigating thread composition problems**

- Finalized reviews and merge of previously started work on string/cat encoder PRs:
  - [https://github.com/scikit-learn/scikit-learn/pull/34386](https://github.com/scikit-learn/scikit-learn/pull/34386) (merged)
  - [https://github.com/scikit-learn/scikit-learn/pull/34392](https://github.com/scikit-learn/scikit-learn/pull/34392) (merged)
  - [https://github.com/scikit-learn/scikit-learn/pull/34393](https://github.com/scikit-learn/scikit-learn/pull/34393) (still WIP)
  - Next PR: improve the complexity of transform for high cardinality values at fit time.
- Root cause analysis of the bad interaction between OpenBLAS and OpenMP on KMeans when run from the scikit-learn [pypi.org](http://pypi.org) wheel packages:
  - [https://github.com/scikit-learn/scikit-learn/issues/34437](https://github.com/scikit-learn/scikit-learn/issues/34437)
- Reviewing discrepancies in available thread limiting semantics:
  - [https://github.com/joblib/threadpoolctl/issues/216](https://github.com/joblib/threadpoolctl/issues/216)
  - [https://github.com/joblib/threadpoolctl/pull/213](https://github.com/joblib/threadpoolctl/pull/213)
- Reported CPU count in joblib when under taskset: [https://github.com/joblib/joblib/issues/1809](https://github.com/joblib/joblib/issues/1809)

**Array API**

- Reviewed improvements to the testing infra for consistent handling of mixed input namespaces, in particular for classification tasks with string labels:
  - [https://github.com/scikit-learn/scikit-learn/pull/32755](https://github.com/scikit-learn/scikit-learn/pull/32755) merged
  - [https://github.com/scikit-learn/scikit-learn/pull/34142](https://github.com/scikit-learn/scikit-learn/pull/34142) merged
  - [https://github.com/scikit-learn/scikit-learn/pull/34439](https://github.com/scikit-learn/scikit-learn/pull/34439) (WIP)
- Finalized the review of the array API support in the newton-cg solver of LogisticRegression
  - [https://github.com/scikit-learn/scikit-learn/pull/34412](https://github.com/scikit-learn/scikit-learn/pull/34412)

**Benchmarks**

- Removed dependency on scikit-learn_bench for flexibility
- Added py-spy traces recording
- Added detailed results table, see for instance: [https://probabl-ai.github.io/scikit-learn-benchmarks/per_hardware.html#intel_laptop_with_b390_gpu](https://probabl-ai.github.io/scikit-learn-benchmarks/per_hardware.html#intel_laptop_with_b390_gpu)
- Almost done: comparing different scikit-learn builds

## 2026-07-01progress meeting

**Investigating thread composition problems**

- [https://github.com/ogrisel/example_pipelines](https://github.com/ogrisel/example_pipelines) (1 BLAS heavy pipeline for now)
- Collected runs on many BLAS setups.
- Summary table: [https://github.com/ogrisel/example_pipelines/blob/main/RESULTS_SUMMARY.md](https://github.com/ogrisel/example_pipelines/blob/main/RESULTS_SUMMARY.md)
  - As expected, oversubscription problems when using Python threads for the outer loop, both with OpenBLAS and MKL (but not Accelerate).
  - Good scalability for Python processes on the outer loop, in particular with MKL and OpenBLAS.
- Goals:
  - Itamar will use that use case to test oversubscription mitigation strategies in threadpoolctl/joblib with the threading backend, in particular with free-threading enabled.
    - [https://github.com/scikit-learn/scikit-learn/issues/34070](https://github.com/scikit-learn/scikit-learn/issues/34070)
    - Itamar and Arthur already started optimizing categorical column encoders.
      - [https://github.com/scikit-learn/scikit-learn/pull/34386](https://github.com/scikit-learn/scikit-learn/pull/34386)
      - [https://github.com/scikit-learn/scikit-learn/pull/34392](https://github.com/scikit-learn/scikit-learn/pull/34392)
      - [https://github.com/scikit-learn/scikit-learn/pull/34393](https://github.com/scikit-learn/scikit-learn/pull/34393)
  - Will also serve as basis for the update Phoronix benchmark proposal.
- Need to add new pipelines for OpenMP heavy workloads, large data to compute ratios.
- Need to explore alternative threading layers for MKL.

**Started work on HGB scaling:**

- Identified first steps to avoid counter-productive scaling:
  - Use prange(..., use_threads_if=...) to disable threading for obviously small workloads where threading overhead is greater than a normal range(...) (WIP)
- Tried & rejected adaptive number of threads strategy

**Benchmarks:**

- Moving forward in implementing goals described in [Probabl x Intel - Technical Scope of Work](https://docs.google.com/document/d/1Mft3NXyzMJKipJscj7JREqlWjvzCOeNcklrQcIMWSnQ/edit?tab=t.lrfjbqz4tkq9)
  - Scalability curves (taskset based)
  - Usability for evaluating a perf PR on scikit-learn
  - Display details of individual runs as a table with sortable and filterable columns
  - Py-spy profiling with links reachable directly from the reports

**Planning**:

- Olivier off between July 15 and July 31.
- Arthur off between July 12 and July 19 (and one week in August)

## 2026-06-24 progress meeting

**GNR being reinstalled**
	Drivers will be already installed: [https://docs.pytorch.org/docs/2.12/notes/get_start_xpu.html](https://docs.pytorch.org/docs/2.12/notes/get_start_xpu.html)
	=> No need for root access
	Probabl users will need to be part of the “render” group to access the GPU.
	SSH access only via keys, no password.

**Scikit-learn benchmark suite ([repo](https://github.com/probabl-ai/scikit-learn-benchmarks%20))**:

- Reworked & improved dashboards:
  - [Software/implementations comparison](https://probabl-ai.github.io/scikit-learn-benchmarks/per_hardware.html)
  - New dashboard: [Hardware comparison](https://probabl-ai.github.io/scikit-learn-benchmarks/hardware_comparisons.html) (a bit hard to read maybe, needs some rework)
- CI to automatically re-build & re-publish dashboards on push
- Draft composed BLAS/LAPACK dominated pipeline with hparam tuning: [https://gist.github.com/ogrisel/1b24301bfc90d61ab2138bb7fbf7f623](https://gist.github.com/ogrisel/1b24301bfc90d61ab2138bb7fbf7f623)
  - Goal is to study CPU scalability with threads or Python processes for the outer loop and interaction between BLAS and OpenMP in various environments.
  - Could also serve as a basis for Phoronix benchmarks.
  - Secondary goal is to study interactions between outer parallelism and GPU parallelism via array API.
  - To be included as a new case for the above benchmarking tool and also used as a tool to drive nested thread parallelism over-subscription protection.

**Hist-gradient boosting work:**

- Minor refactor: [https://github.com/scikit-learn/scikit-learn/pull/34335](https://github.com/scikit-learn/scikit-learn/pull/34335) (merged)
- Stronger testing: [https://github.com/scikit-learn/scikit-learn/pull/34343](https://github.com/scikit-learn/scikit-learn/pull/34343) (merged)
- Performance improvements:
  - [https://github.com/scikit-learn/scikit-learn/pull/34194](https://github.com/scikit-learn/scikit-learn/pull/34194) (merged)
  - [https://github.com/scikit-learn/scikit-learn/pull/34248](https://github.com/scikit-learn/scikit-learn/pull/34248)
  - [https://github.com/scikit-learn/scikit-learn/pull/34381](https://github.com/scikit-learn/scikit-learn/pull/34381)
- Some on-the-side benchmarking (long-term: to be merged within the scikit-learn benchmark suite): [https://github.com/cakedev0/numerical-python-bench/blob/main/results/hgbt_sweep/summary.txt](https://github.com/cakedev0/numerical-python-bench/blob/main/results/hgbt_sweep/summary.txt)
  - Clearly identifies what doesn’t scale: finding & applying splits mostly
  - Likely culprit: [prange start-up time](https://github.com/cakedev0/numerical-python-bench/blob/main/results/laptop_prange_startup.txt) over small workloads:
    - Finding a split is O(n_bins x n_features_for_split) => can be small
    - Applying a split is O(n_samples) => can be small for small datasets or big trees

## 2026-06-17 progress meeting

**Scikit-learn benchmark suite**:

- [https://github.com/probabl-ai/scikit-learn-benchmarks](https://github.com/probabl-ai/scikit-learn-benchmarks)
- Configurable reporting logic that does not require rerunning the code to collect the measurements.
  - Reworking the code for better control (vibe-coding moves fast but rapidly explodes in an unauditable codebase)
  - Hardware and software (versions) description in the report.
- Figuring out small details about fairness, what can be compared, how to inform users of certain mismatch (ex: max_bins impact, different solvers/number of iterations, different scores, etc.)
  - For instance, I changed some benchmark data from the default F-order to C-order to accommodate sklearnex (see [this issue](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3235)): read_csv/parquet + py-arrow => F-order
    - See what results we get with typical transformer
    - Short term: do both? Then switch to most likely
- Possible feature: Python code generation.
  - Purpose: make it easier to run the script of particular case to iteratively investigate by editing the code to instrument it.
  - Purpose: make it possible to run the benchmark for individual cases without installing the benchmark framework itself (e.g. for phoronix).
  - What do you think?

**Python level threading performance impact on scikit-learn scalability**:

- Ongoing discussion to evolve the threadpoolctl inspection and threadpool limiting API to be able to cleanly address heterogeneity in threadpool semantics of different runtimes: [https://github.com/joblib/threadpoolctl/issues/208](https://github.com/joblib/threadpoolctl/issues/208)

**Phoronix benchmark:** started auditing current benchmarks.

- It uses old unmaintained scikit-learn 1.2.2 benchmarks that in some cases scale extremely poorly (e.g. up to 20x slowdown from a tiger lake laptop to a 64-cores EPYC server)...  for histogram gradient boosting
- Compile everything from source to compare different compilers and different compiler flags.
  - -march native and link time optimization.
- Make sure that the benchmark code reports runtime information such as the BLAS implementation, number of threads…

**Planned:**

- Make it possible to explicitly configure pixi environment for each benchmark case
- audit the existing performance guidelines in the scikit-learn documentation
  - [https://scikit-learn.org/stable/computing/computational_performance.html](https://scikit-learn.org/stable/computing/computational_performance.html)
    - Intel analog: [https://github.com/intel/optimization-zone/tree/main/software/scikit-learn](https://github.com/intel/optimization-zone/tree/main/software/scikit-learn)
  - [https://scikit-learn.org/stable/computing/parallelism.html](https://scikit-learn.org/stable/computing/parallelism.html)
- Investigate the cause of scalability issues of hist-gradient boosting classifier on many cores systems (e.g. the GNR once available).
- write canonical ML pipelines that could be used as official benchmark scripts to assess performance impacts of different choice hardware/runtime/threading layers.
  - In particular, included composition of components with nested parallelism (e.g. cross-validation, grid search, permutation importance) that better reflect the actual user experience in performance sensitive settings.

## 2026-06-10 progress meeting (cancelled)

Scikit-learn 1.9.0 released with official Intel GPU support via array API

Ongoing scikit-learn sprint:

- Coordinating on roadmap priorities, array API default policy dispatching, strategies for hardware acceleration beyond array API, tree models improvements, possible use of C++ within the scikit-learn code base.

## 2026-06-03 progress meeting

Fixes/reports in scikit-learn-intelex/oneDAL:

- Fixes in oneDAL trees: [https://github.com/uxlfoundation/oneDAL/pull/3649](https://github.com/uxlfoundation/oneDAL/pull/3649) (3 fixes, new regression tests)
- Upstream report for oneDAL: [https://github.com/uxlfoundation/scikit-learn-intelex/pull/3224#issuecomment-4591685801](https://github.com/uxlfoundation/scikit-learn-intelex/pull/3224#issuecomment-4591685801)
- [F-order dpnp arrays make Ridge GPU array API extremely slow](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3235)
- Sklearnex issues:
  - [LogisticRegression GPU fallback ignores allow_sklearn_after_onedal=False](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3220)
  - [scikit-learn-intelex-gpu is documented for conda-forge but missing from the channel](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3211)

Array API fixes/issues:

- Sparse investigation: [https://github.com/scikit-learn/scikit-learn/issues/34087#issuecomment-4593841317](https://github.com/scikit-learn/scikit-learn/issues/34087#issuecomment-4593841317)
  - => create a follow-up informational issue? Variant/Extension to array API?
  - Related: [https://github.com/data-apis/array-api/issues/840](https://github.com/data-apis/array-api/issues/840)

Benchmarks:

- Repo: [https://github.com/probabl-ai/scikit-learn-benchmarks](https://github.com/probabl-ai/scikit-learn-benchmarks) (private for now, while I haven’t configured the CI).
- First results: [https://probabl-ai.github.io/scikit-learn-benchmarks/](https://probabl-ai.github.io/scikit-learn-benchmarks/) (very WIP still, a lot of things to double check & analyze, fast runs, running on the laptop for now)
  - [https://probabl-ai.github.io/scikit-learn-benchmarks/nvidia.html](https://probabl-ai.github.io/scikit-learn-benchmarks/nvidia.html)
  - -> comment here: check for convergence in logistic regression. Oftentimes reaches max iter without converging
- Hardware:
  - Set-up a scaleway instance with nvidia L4 GPUs (only running array-API with cuda backend for now)
  - Set-up GNR:
    - Lock based sharing with UVSQ
    - not seeing the GPU for now
- [discussed during monthly call] Efficiency/Energy usage? [https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Anaconda-Demo-Greener-Machine-Learning-with-Intel/post/1500060](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Anaconda-Demo-Greener-Machine-Learning-with-Intel/post/1500060) - pmi data (power consumption for CPU, GPU should have similar things: [https://github.com/intel/pti-gpu](https://github.com/intel/pti-gpu)). Phoronix does power consumption & total cost. [GitHub - szilard/GBM-perf: Performance of various open source GBM implementations](https://github.com/szilard/GBM-perf). They also track interesting historical data points.

Threading oversubscription mitigation:

- [https://github.com/scikit-learn/scikit-learn/pull/34171](https://github.com/scikit-learn/scikit-learn/pull/34171)
- WIP: crafting some high-level scikit-learn pipelines / with either BLAS or OpenMP or dominated workloads or mixed runtimes.

For problems with setup:

- TODO for olivier report issue under mkl-service repository

TODO: compilation flags?

## 2026-05-27 progress meeting

- Array API fixes/issues:
  - RidgeClassifier on mixed inputs [https://github.com/scikit-learn/scikit-learn/pull/34065](https://github.com/scikit-learn/scikit-learn/pull/34065): merged
  - Discussion started on mixed inputs for classifiers/classification scores: [https://github.com/scikit-learn/scikit-learn/issues/33822](https://github.com/scikit-learn/scikit-learn/issues/33822) & [https://github.com/scikit-learn/scikit-learn/issues/34094](https://github.com/scikit-learn/scikit-learn/issues/34094)
  - Discussion started for non-scipy sparse handling: [https://github.com/scikit-learn/scikit-learn/issues/34087](https://github.com/scikit-learn/scikit-learn/issues/34087)
  - Discussion started about public array API functions [https://github.com/scikit-learn/scikit-learn/issues/34135](https://github.com/scikit-learn/scikit-learn/issues/34135)

-  Array API documentation: [https://github.com/scikit-learn/scikit-learn/pull/34054](https://github.com/scikit-learn/scikit-learn/pull/34054) (still WIP, close to be merged, should be backported in 1.9)
- CI/Benchmarks infrastructure: discussion with Pablo de Oliveira Castro for the shared server with UVSQ
- Benchmarks:
  - Repo still private for now, while I set-up the CI
  - Forked [scikit-learn_bench](https://github.com/IntelPython/scikit-learn_bench/tree/main/sklbench)
    - Started customization on branch “probabl-fork”: [https://github.com/cakedev0/scikit-learn_bench/tree/probabl-fork](https://github.com/cakedev0/scikit-learn_bench/tree/probabl-fork)
    - Upstream contribution: [small fix for pandas 3 compat](https://github.com/IntelPython/scikit-learn_bench/pull/213)
  - Adapted to work with pixi: is the contribution welcomed upstream?
  - Adapted to allow easier “partial cartesian product”, to allow easily defining a parameters set with [

    {**many_params, “a”: 0, “b”: 0},

    {**many_params, “a”: 1, “b”: 0},

    {**many_params, “a”: 0, “b”: 1}

    ]

- Bug report on oneDAL: [https://github.com/uxlfoundation/oneDAL/issues/3648](https://github.com/uxlfoundation/oneDAL/issues/3648)
  - Draft PR: [https://github.com/uxlfoundation/oneDAL/pull/3649](https://github.com/uxlfoundation/oneDAL/pull/3649)
- Ongoing discussions on Python level thread-safety testing: [https://github.com/scikit-learn/scikit-learn/issues/34072](https://github.com/scikit-learn/scikit-learn/issues/34072)
- Ongoing discussion on oversubscription between Python level thread parallelism and nested OpenMP/OpenBLAS parallelism:
  - [https://github.com/scikit-learn/scikit-learn/issues/34070](https://github.com/scikit-learn/scikit-learn/issues/34070)
- Threaded libraries packaging problems: on Linux libgomp and openblas do not like each (concurrent active wait) other but libomp is fine (once E cores and LP cores are ignored):
  - [https://github.com/scikit-learn/scikit-learn/issues/17334](https://github.com/scikit-learn/scikit-learn/issues/17334)
  - Two possible fixes:
    - Avoid E/LP core related oversubscription: nullifies the overhead of the active wait interactions for some reason.
    - Fix the pypi packaging to disable active wait in OpenBLAS or switch from libgomp to libomp for PyPI packages
    - Alternative: make it possible for SciPy to ship a BLAS implementation that uses a non-conflicting OpenMP, either via wheelnext or a change in packaging practices.

## 2026-05-20 progress meeting

- Array API fixes:
  - RidgeClassifier on mixed inputs [https://github.com/scikit-learn/scikit-learn/pull/34065](https://github.com/scikit-learn/scikit-learn/pull/34065) (WIP)
  - Related paragraph: [https://scikit-learn.org/dev/modules/array_api.html#input-and-output-array-type-handling](https://scikit-learn.org/dev/modules/array_api.html#input-and-output-array-type-handling) but does not cover the string class label case explicitly.
  - TODO: make string label handling explicit in the doc.
- Array API documentation mentions Intel GPU support:
  - [https://scikit-learn.org/dev/modules/array_api.html](https://scikit-learn.org/dev/modules/array_api.html)
  - [https://github.com/scikit-learn/scikit-learn/pull/34021](https://github.com/scikit-learn/scikit-learn/pull/34021) (merged)
  - [https://github.com/scikit-learn/scikit-learn/pull/34054](https://github.com/scikit-learn/scikit-learn/pull/34054) (WIP follow-up refactoring)
- Logistic regression and multithreading:
  - [https://github.com/scikit-learn/scikit-learn/issues/32162](https://github.com/scikit-learn/scikit-learn/issues/32162) (still under discussion)
- Improvement to the CI infrastructure.
  - Security:
    - Locked dependencies with pixi (but not perfect because of an issue in pytorch index, encounters difficulties were reported upstream: [https://github.com/pytorch/pytorch/issues/179374#issuecomment-4467404210](https://github.com/pytorch/pytorch/issues/179374#issuecomment-4467404210)
    - Isolated unix user for the runner
  - Example of use to test on a scikit-learn PR: [https://github.com/scikit-learn/scikit-learn/pull/33573#issuecomment-4489794306](https://github.com/scikit-learn/scikit-learn/pull/33573#issuecomment-4489794306)
  - But still things to improve.
- Benchmarks plans:
  - Start with a user-facing benchmarks based on Intel’s [scikit-learn_bench](https://github.com/IntelPython/scikit-learn_bench/tree/main/sklbench)
  - Make our fork of [scikit-learn_bench](https://github.com/IntelPython/scikit-learn_bench/tree/main/sklbench) to customize some things if needed (for instance, we might want to be able to benchmark sklearn pipelines, and not just a single model)
  - Make another repo with:
    - CI workflows
    - Configs
    - Persisted results for common scikit-learn estimators and few pipelines
    - User-facing “dashboard” (it could be as simple as a readme with plots, or could be an interactive html hosted on github page) that will serve as a base for communication
  - Goal: measure speed-up of sklearnex/sklearn+array-API/(rapids cuML?)/xgboost/lightgbm vs sklearn in a way that will be meaningful to communicate
      The goal of guiding optimization work in scikit-learn will remain secondary for now.
    - Focus first on easy cases where there exists an exact hyperparameter matching.
    - Need to investigate how to set hparams to be able to compare scikit-learn-intelex RFs with scikit-learn vanilla RFs that do not yet support histograms on binned features. Could include xgboost RFs to the mix.

- Python-level thread safety analysis in the context of free-threading Python: [https://github.com/scikit-learn/scikit-learn/pull/34057](https://github.com/scikit-learn/scikit-learn/pull/34057) (WIP)
- Cross-platform hybrid CPU inspection (WIP):
  - [https://github.com/joblib/loky/pull/630](https://github.com/joblib/loky/pull/630)

## 2026-05-13 progress meeting

- Technical progress:
  - Scikit-learn
    - PR approved: [FIX: Array API: Fix CPU-dpnp tests](https://github.com/scikit-learn/scikit-learn/pull/33898) (waiting for a second reviewer)
    - PR merged: [TST: Fix flakyness of array-API Poisson regression compliance test](https://github.com/scikit-learn/scikit-learn/pull/33929)
  - PyTorch XPU: [https://github.com/pytorch/pytorch/issues/182282](https://github.com/pytorch/pytorch/issues/182282) => closed because it was already fixed a couple of weeks ago…
    - TODO: add entries to the CI to also run the scikit-learn tests against the PyTorch XPU and dpnp nightly builds (either using custom pip indices or conda label channels). pip install -i [https://pypi.anaconda.org/dppy/simple](https://pypi.anaconda.org/dppy/simple) dpnp.
    - [https://anaconda.org/channels/dppy/packages/dpnp/overview](https://anaconda.org/channels/dppy/packages/dpnp/overview)
    - TODO: As soon as we have green CI on the PyTorch XPU dev, let’s open a PR on the scikit-learn documentation to state that Intel GPUs are officially supported (and link to installation instructions in DPNP and PyTorch XPU).
  - Started to work on benchmarks specs:
    - Manually compared sklearnex RF vs sklearn RF:
      - Will drive some benchmarks design decision (careful HP matching, output comparison, e.g. tree structure)
      - Similar performance when using the same algorithm (max_bins=n_samples), but hist-based splitting (sklearnex default) is much faster
    - Reviewing
      - [https://github.com/mbatoul/sklearn_benchmarks](https://github.com/mbatoul/sklearn_benchmarks):
        - Nice reporting ideas
        - Randomized HPO benchmarks
      - [https://github.com/IntelPython/scikit-learn_bench](https://github.com/IntelPython/scikit-learn_bench): WIP
    - Choosing profiling tool: py-spy vs samply vs VTune?
    - First few algorithms: RF, Ridge, LogisticRegression
  - WIP: started to investigate scalability of scikit-learn HistGradientBoostingRegressor
    - Scalability is ok on large dataset (compared to lightgbm and xgboost)
    - R2 score is slightly lower despite some efforts to set identical hparam values
    - Scalability of scikit-learn is poor (compared to lightgbm) on smaller problems.
    - Efficiency of scikit-learn is catastrophic in oversubscribed settings (worse than lightgbm which is itself worse than xgboost).
    - Opened issue to discuss detection https://github.com/joblib/loky/issues/629
- SSH Server access:
  - David shared an IP address in a discord chat but it’s not clear if it was meant for sharing with Probabl engineers or not.
  - Nikolay stated that a different machine (workstation) is still being provisioned / waiting for network config).

## 2026-05-06 progress meeting

- GOSIM presentations
- Technical progress:
  - Scikit-learn
    - Intel GPU CI configuration: [https://github.com/probabl-ai/scikit-learn-intel-workflow/actions](https://github.com/probabl-ai/scikit-learn-intel-workflow/actions)
    - PR: [FIX: Array API: Fix CPU-dpnp tests](https://github.com/scikit-learn/scikit-learn/pull/33898)
    - PR: [TST: Fix flakyness of array-API poisson regression compliance test](https://github.com/scikit-learn/scikit-learn/pull/33929)
  - PyTorch XPU
    - float64 investigation lead to discovering bad precision for mat-vec: [https://github.com/pytorch/pytorch/issues/182282](https://github.com/pytorch/pytorch/issues/182282)

                => no unexplained/unfixed test failures remaining

- array-api-strict
  - Reported improvement requests based on limitations discovered by DPNP test failures (that are now fixed on the scikit-learn side):
    - [https://github.com/data-apis/array-api-strict/issues/207](https://github.com/data-apis/array-api-strict/issues/207)
      - SPEC refinement: [https://github.com/data-apis/array-api/pull/1005](https://github.com/data-apis/array-api/pull/1005)
- Discord setup with invite link.
- Meeting frequency:
  - Big meeting every month.
  - Shorter technical meeting every week.
    - On discord.
    - Olivier to create to invite.
  - Same time.

## 2026-04-29 progress meeting

- [Arthur Lacote](mailto:arthur.lacote@probabl.ai) joined the project as an engineer to help with the implementation of the points document in the SoW document.
- Hardware status:
  - Laptop with Intel® Core™ Ultra X7 Processor 358H iGPU (Arc B390) provisioned and shared by Arthur and Olivier via SSH access for interactive dev/debug purposes.
  - On hold: B60 discrete GPU available but probabl would need to find and setup a host.
  - In progress: work to host a machine with remote access at Intel.
- Communication:
  - Share some quick array API benchmark numbers on public hardware during  Olivier number at GOSIM.
  - No official statement for the collaboration between Intel and Probabl at GOSIM, but we can say that we already collaborate.
  - Main announcement to be done at Vivatech.
  - Can have lunch together at Station F during GOSIM.
- Technical progress:
  - Scikit-learn:
    - First PR to add DPNP and PyTorch XPU support in scikit-learn array API testing infra: [https://github.com/scikit-learn/scikit-learn/pull/32460](https://github.com/scikit-learn/scikit-learn/pull/32460) (merged)
  - PyTorch (XPU backend):
    - [https://github.com/pytorch/pytorch/issues/181140](https://github.com/pytorch/pytorch/issues/181140) (issue with reproducer by Olivier)
    - [https://github.com/pytorch/pytorch/pull/181361](https://github.com/pytorch/pytorch/pull/181361) (fix by upstream maintainer merged)
  - array-api-strict
    - Reported a minimal reproducer for a weakness in this testing tool:
      - [https://github.com/data-apis/array-api-strict/issues/70#issuecomment-4296452855](https://github.com/data-apis/array-api-strict/issues/70#issuecomment-4296452855)
      - Potential fix by upstream maintainer: [https://github.com/data-apis/array-api-strict/pull/206](https://github.com/data-apis/array-api-strict/pull/206) (WIP)
