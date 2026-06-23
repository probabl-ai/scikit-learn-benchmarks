# scikit-learn-benchmarks

Work-in-progress benchmark configuration for comparing scikit-learn-compatible
machine learning libraries and array API backends.

This repository is currently exploratory. Configurations, result formats, and
reporting scripts may change without compatibility guarantees.

## Setup

```bash
git submodule update --init --recursive
```

Install Git LFS before checking out or adding benchmark results. On
Ubuntu/Debian:

```bash
apt update
apt install git-lfs
git lfs install
git lfs pull
```

## Basic checks

Validate the local benchmark configuration:

```bash
pixi run python validate_config.py configs/sklearn.json
```

Preview the benchmark cases that would be expanded from the configs:

```bash
pixi run python preview_cases.py configs/sklearn.json trees --count
```

## Running benchmarks

Run the default scikit-learn configuration:

```bash
pixi run -e sklearn python -m sklbench --config configs/sklearn.json
```

For Intel and array API benchmark work, use the `intel` pixi environment:

```bash
pixi run -e intel ./test.sh
```

Generated benchmark results are written as flat `results/<timestamp>.json` files.
Captured hardware/software environments are stored under `results/hardware-envs/`
and `results/software-envs/` using hash-only filenames. JSON files under
`results/` are tracked with Git LFS; run `git lfs install` before adding or
checking out benchmark results.

## Status

Known next steps include: writing a proper TODO
