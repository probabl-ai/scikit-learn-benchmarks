# scikit-learn Benchmarks

This repository publishes benchmark results for scikit-learn and compatible
implementations across CPU, GPU, and Array API backends.

The goal is to make performance trade-offs visible for scikit-learn users:
which workloads benefit from alternative hardware or backends, which results are
comparable to the scikit-learn baseline, and where fallback behavior or metric
differences require caution.

## Dashboards

The latest generated dashboards are published on GitHub Pages:

- [Dashboard index](https://probabl-ai.github.io/scikit-learn-benchmarks/)
- [Hardware comparison](https://probabl-ai.github.io/scikit-learn-benchmarks/hardware_comparisons.html)
- [Software/implementations comparison](https://probabl-ai.github.io/scikit-learn-benchmarks/per_hardware.html)

## Reading the Results

The dashboards compare matched benchmark cases and report speed-ups relative to
a scikit-learn baseline. Hover over points to inspect the estimator, dataset
shape, timings, warnings, and metric differences for a specific case.

Warnings call out important comparison details such as different iteration
counts, histogram-based tree behavior, CPU fallback, or metric differences.
Those cases are still useful, but they should not be read as strict
like-for-like speed comparisons without checking the warning details.

## Scope

The benchmark suite currently focuses on:

- linear models
- tree-based models
- clustering algorithms
- scikit-learn, scikit-learn-intelex, and Array API backends
- selected CPU, Intel GPU, and NVIDIA GPU environments

> **Warning**
> This project is still exploratory. Configurations, result formats, and
> dashboards may change as the benchmark coverage matures.

## Contributing

Developer setup, architecture notes, and instructions for adding benchmark
cases or publishing new benchmark results live in [CONTRIBUTING.md](CONTRIBUTING.md).

Instructions for triggering an automated before/after benchmark comparison
against an upstream scikit-learn PR live in [COMPARISONS_PR.md](COMPARISONS_PR.md).
