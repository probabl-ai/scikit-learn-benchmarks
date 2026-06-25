# Results Failures Analysis

Analysis of `results/` as of the current checkout.

## Overview

- Successful benchmark cases: 637
- Failed benchmark cases: 32
- All hard failures are in `results/20260622T162516419110Z.json`.
- The failing run is the `intel` pixi environment on Intel Arc B390 hardware.
- Other result files have no `failed_cases`; they only contain warnings or informational logs.

## Address for Fair Benchmarking

### sklearnex fallback failures

These failures are expected from the current fairness guard.

`configs/_common.py` sets both of these to `false` for sklearnex:

- `allow_fallback_to_host`
- `allow_sklearn_after_onedal`

The benchmark harness then raises when sklearnex falls back to original scikit-learn. This is the right behavior for an accelerated sklearnex comparison, because otherwise timings could silently include host-side scikit-learn work.

Failed fallback matrix:

| Implementation | Estimator | Cases | Handling |
| --- | --- | ---: | --- |
| `sklearnex/cpu` | `LogisticRegression(solver="newton-cg")` | 4 | Exclude from accelerated sklearnex comparison |
| `sklearnex/cpu` | `Ridge(solver="svd")` | 4 | Exclude from accelerated sklearnex comparison |
| `sklearnex/gpu/dpnp` | `LogisticRegression(solver="lbfgs")` | 4 | Exclude from accelerated sklearnex comparison |
| `sklearnex/gpu/dpnp` | default `LogisticRegression` | 4 | Exclude from accelerated sklearnex comparison |
| `sklearnex/gpu/dpnp` | `Ridge(solver="svd")` | 4 | Exclude from accelerated sklearnex comparison |

Recommendation: keep these as unsupported or failed for the accelerated sklearnex view. Only include them in a separate “sklearnex with scikit-learn fallback allowed” category if compatibility behavior is explicitly being benchmarked.

### Unsupported sklearnex estimator

`RidgeClassifier(solver="svd")` fails under sklearnex with:

```text
ValueError: Unable to find RidgeClassifier estimator in sklearnex module.
```

This appears 8 times:

- 4 CPU cases
- 4 GPU/dpnp cases

Recommendation: filter `RidgeClassifier` out of the sklearnex matrix rather than treating these as runtime failures.

### Intel Arc sklearnex GPU clustering failures

The sklearnex GPU/dpnp clustering failures need separate handling before using those comparisons.

| Estimator | Dataset / size | Error | Handling |
| --- | --- | --- | --- |
| `KMeans` | `make_blobs`, `500000 x 3` | `UR_RESULT_ERROR_DEVICE_LOST` | Rerun after device reset; do not compare as complete |
| `KMeans` | MNIST, train `50000` | `UR_RESULT_ERROR_OUT_OF_RESOURCES` | Scale down or mark as resource-limited |
| `KMeans` | MNIST, train `10000`, `n_clusters=100` | timeout after 25s | Increase timeout or mark as too slow |
| `DBSCAN` | `make_blobs`, `50000 x 3` | `UR_RESULT_ERROR_DEVICE_LOST` | Rerun after device reset; do not compare as complete |

Recommendation: mark these as Intel Arc B390 sklearnex GPU runtime/resource failures. Avoid comparing missing sklearnex GPU clustering entries against successful CPU/sklearn rows as if the matrix is complete.

### PyTorch XPU CPU op fallback

`results/20260622T161759679144Z.json` contains successful `sklearn/xpu/torch` cases with:

```text
Aten Op fallback from XPU to CPU
```

This affects:

- `Ridge`: 4 cases
- `RidgeClassifier`: 4 cases

The warning comes from `xp.linalg.svd`, so these should not be described as pure XPU acceleration.

Recommendation: keep the results, but label them as containing PyTorch XPU CPU fallback, or exclude them from a strict accelerator-only comparison.

## Mostly Expected / Low Priority

### LogisticRegression convergence warnings

There are 15 successful cases with scikit-learn `ConvergenceWarning` from `LogisticRegression(max_iter=100)`.

Recommendation: acceptable for same-config timing comparisons. If quality metrics are important, increase `max_iter` or document that convergence is not guaranteed.

### Measurement time limit warnings

There are 5 successful cases where measurement exceeded the configured `time_limit`.

Affected estimators include:

- `Ridge`
- `RidgeClassifier`

Recommendation: low priority unless uniform `n_runs=5` is required. If so, raise the `time_limit`.

## Bottom Line

The sklearnex fallback errors are expected and useful: they show that the fairness guard is preventing silent fallback to original scikit-learn. The main cleanup is to filter unsupported sklearnex estimator, solver, and device combinations before running or reporting, and to mark Intel Arc sklearnex GPU clustering failures separately from valid benchmark results.
