# Contributing

This repository contains benchmark configurations, captured benchmark results,
and reporting scripts for the published scikit-learn benchmark dashboards.

## Setup

Initialize the nested benchmark runner checkout:

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

The project uses Pixi environments. Common environments are:

- `sklearn`: vanilla scikit-learn benchmark runs
- `skl-cpu`: Array API CPU benchmark runs
- `skl-intel`: Intel Array API benchmark runs
- `skl-nvidia`: NVIDIA Array API benchmark runs
- `intel`: scikit-learn-intelex CPU/GPU benchmark runs
- `reporting`: dashboard generation and reporting utilities

## Architecture

The repository is split into a few layers:

- `configs/`: Python benchmark case generators. Top-level config scripts
  combine model/data helpers with implementation definitions and expose
  `generate_cases()`.
- `scikit-learn_bench/`: nested checkout of the benchmark runner.
- `sklbench`: symlink to `scikit-learn_bench/sklbench`.
- `results/`: captured benchmark outputs and environment metadata, tracked
  with Git LFS.
- `reporting/`: result matching, environment summaries, and HTML helpers.
- `dashboard/gen_*.py`: dashboard entry points. Each script reads `results/`
  and writes one HTML page.
- `.github/workflows/dashboard-pages.yml`: GitHub Pages workflow. It runs all
  `dashboard/gen_*.py` scripts on pushes to `main` and publishes the generated
  HTML.

Config scripts are regular Python. They return a list of JSON-serializable case
dictionaries from `generate_cases()`, and the orchestrator validates each case
before running it. Shared helpers live in `configs/_common.py`.

## Adding Benchmark Cases

Most case changes should start in `configs/_common.py`.

Use the default `test` template for the current small exploratory matrix. Set
`SKBENCH_MODELS_TEMPLATE=fast` when working on a broader but still reasonably
fast matrix.

Preview and validate a config by importing it directly:

```bash
pixi run python - <<'PY'
from sklbench.config import load_cases_from_script

cases = load_cases_from_script("configs/sklearn.py")
print(len(cases))
print(cases[0])
PY
```

When adding cases, keep the matrix small enough to run repeatedly, set
deterministic estimator/data random states where relevant, and check that the
case can be compared to a baseline by the dashboard matching logic.

## Running Benchmarks

Run the default scikit-learn configuration:

```bash
pixi run -e sklearn python -m sklbench --config configs/sklearn.py
```

Run the CPU benchmark set:

```bash
./run.sh
```

Run CPU plus Intel GPU benchmarks:

```bash
./run.sh intel
```

Run CPU plus NVIDIA GPU benchmarks:

```bash
./run.sh nvidia
```

`run.sh` expands to the appropriate Pixi environments and config files for
each mode. For ad hoc Intel or reporting work, prefer the explicit environments:

```bash
pixi run -e intel ...
pixi run -e reporting ...
```

Generated benchmark results are written as flat `results/<timestamp>.json`
files. Captured hardware and software environments are stored under
`results/hardware-envs/` and `results/software-envs/` using hash-only filenames.

## Previewing Dashboards Locally

Generate all dashboard pages into a temporary directory:

```bash
mkdir -p /tmp/sklbench-dashboard
for script in dashboard/gen_*.py; do
  pixi run -e reporting python "$script" --output-dir /tmp/sklbench-dashboard
done
```

Open `/tmp/sklbench-dashboard/index.html` in a browser to inspect the local
output.

During dashboard development, use the watcher to regenerate pages whenever
`results/`, `reporting/`, or `dashboard/` changes:

```bash
pixi run -e reporting python watch_dashboards.py --output-dir /tmp/sklbench-dashboard
```

## Publishing New Results

Before committing results:

```bash
git lfs install
git lfs pull
```

Run the relevant benchmarks, then inspect the generated files:

```bash
git status --short results/
```

Stage the new result and environment JSON files together:

```bash
git add results/
```

If the benchmark configuration changed, stage the matching config changes too:

```bash
git add configs/
```

Commit and push the change through the normal repository workflow. Once the
change reaches `main`, the GitHub Pages workflow regenerates and deploys the
dashboards automatically.

## Notes

`results/*.json` and `results/**/*.json` are tracked through Git LFS via
`.gitattributes`. Do not bypass LFS for benchmark results.

Changes under `sklbench/` belong to the nested `scikit-learn_bench` checkout,
not this top-level repository.
