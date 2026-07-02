# Contributing

This repository contains benchmark configurations, captured benchmark results,
and reporting scripts for the published scikit-learn benchmark dashboards.

## Setup

Install the Pixi environments from the repository root. The local `sklbench`
package is installed editable by Pixi.

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
- `sklbench/`: local benchmark package containing config models, orchestrator,
  runners, runner datasets, and reporting helpers.
- `results/`: captured benchmark outputs and environment metadata, tracked
  with Git LFS.
- `sklbench/reporting/`: result matching, environment summaries, and HTML helpers.
- `dashboards/gen_*.py`: dashboard entry points. Each script reads `results/`
  and writes one HTML page.
- `.github/workflows/dashboard-pages.yml`: GitHub Pages workflow. It runs all
  `dashboards/gen_*.py` scripts on pushes to `main` and publishes the generated
  HTML.

Config scripts are regular Python. They return a list of JSON-serializable case
dictionaries or pydantic case models from `generate_cases()`, and the
orchestrator validates each case before running it. Shared helpers live in
`configs/_generators.py`.

## Adding Benchmark Cases

Most case changes should start in `configs/_generators.py`.

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

Generated benchmark records are written under `results/records/`. Captured
hardware and software environments are stored under `results/hardware-envs/`
and `results/software-envs/` using hash-only filenames.

**Cloud machines can have high tail variability**, especially for scaling studies
and short workloads. Prefer stable local/dedicated hardware when deciding whether
one representative case can replace a broader matrix.

## Running Against scikit-learn Branches

For local performance PR checks, use the development Pixi environment and the
scikit-learn setup helper. It fetches the requested scikit-learn ref into this
repo's managed git cache under `.bench/`, creates a detached worktree, and
installs that checkout editable into `sklearn-dev`.

```bash
scripts/setup_sklearn_ref.sh \
  --ref main \
  --label main

pixi run -e sklearn-dev python -m sklbench --config configs/sklearn.py
```

Run the same config against a branch from a fork by changing only the remote and
ref:

```bash
scripts/setup_sklearn_ref.sh \
  --remote https://github.com/some-user/scikit-learn.git \
  --ref my-perf-branch \
  --label pr

pixi run -e sklearn-dev python -m sklbench --config configs/sklearn.py
```

Pass orchestrator arguments directly to `sklbench` in the second command, for
example `--results-dir /tmp/sklbench-results` or `--exit-on-error`.

For local scikit-learn edits, make a temporary commit in the scikit-learn
checkout and use that checkout as the remote. Benchmark results are meant to be
identified by a commit SHA; uncommitted edits are intentionally not a supported
input.

```bash
cd ~/src/scikit-learn
git switch -c bench/my-change
git add sklearn/path/to/file.py
git commit -m "WIP benchmark change"

cd -
scripts/setup_sklearn_ref.sh \
  --remote ~/src/scikit-learn \
  --ref bench/my-change \
  --label wip
```

The software environment JSON records runtime import metadata for packages such
as `sklearn`, including git commit information when the imported package comes
from a git checkout.

## Previewing Dashboards Locally

Generate all dashboard pages into a temporary directory:

```bash
mkdir -p /tmp/sklbench-dashboard
for script in dashboards/gen_*.py; do
  pixi run -e reporting python "$script" --output-dir /tmp/sklbench-dashboard
done
```

Open `/tmp/sklbench-dashboard/index.html` in a browser to inspect the local
output.

During dashboard development, use the watcher to regenerate pages whenever
`results/`, `sklbench/reporting/`, or `dashboards/` changes:

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

Changes under `sklbench/` are part of this repository.
