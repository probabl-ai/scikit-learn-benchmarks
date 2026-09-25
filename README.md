# scikit-learn Benchmarks

This repository publishes benchmark results for scikit-learn and compatible
implementations, on CPU, GPU and Array API backends.

The goal is to show scikit-learn users the performance trade-offs: which
workloads benefit from other hardware or backends, which results are
comparable to the scikit-learn baseline, and when fallbacks or metric
differences call for caution.

## Dashboards

Dashboards are published on GitHub Pages:

- [Dev dashboard](https://probabl-ai.github.io/scikit-learn-benchmarks/):
  regenerated on every push to `main`, so it always has the latest results.
- [Latest stable dashboard](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/2026-09-02-cff6ea4/):
  a snapshot of a specific commit, published by the `Dashboard Snapshot`
  GitHub Actions workflow.

## Reading the results

The dashboards match benchmark cases across implementations and report
speed-ups over a scikit-learn baseline. Hover a point to see the estimator,
dataset shape, timings, warnings and metric differences of that case.

Warnings flag comparison details such as different iteration counts,
histogram-based trees, CPU fallback or metric differences. These cases are
still useful, but check the warning before reading them as like-for-like
speed comparisons.

## Scope

The benchmarks currently cover:

- linear models
- tree-based models
- clustering algorithms
- scikit-learn, scikit-learn-intelex and Array API backends
- a selection of CPU, Intel GPU and NVIDIA GPU environments

> **Warning**
> This project is still exploratory. Configs, result formats and dashboards
> may change as coverage grows.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the developer setup, the
architecture, and how to add benchmark cases or publish results.

See [COMPARISONS_PR.md](COMPARISONS_PR.md) to benchmark an upstream
scikit-learn PR against `main` automatically.
