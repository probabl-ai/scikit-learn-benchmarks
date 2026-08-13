# Contributing

This repository contains benchmark configurations, captured benchmark results,
and reporting scripts for the published scikit-learn benchmark dashboards.

## Setup

Install the Pixi environments from the repository root. The local `sklbench`
package is installed editable by Pixi.

Set `PIXI_FROZEN=true` in your shell (e.g. `export PIXI_FROZEN=true` in your
profile) before running any `pixi` command in this repo. The `sklearn-dev`
environment (see "Running Against scikit-learn Branches" below) depends on a
local path (`sklearn-src/`) that usually doesn't exist yet on a fresh clone;
without `PIXI_FROZEN=true`, pixi's default lockfile-freshness check
canonicalizes that path for *every* `pixi run` regardless of `-e`, so any
environment fails until `scripts/setup_sklearn_ref.sh` has been run at least
once. `PIXI_FROZEN=true` skips that check and trusts `pixi.lock` as-is - which
also means a `pixi.toml` dependency change won't take effect until you
explicitly run `pixi lock` or `pixi install`.

Install Git LFS before checking out or adding benchmark results. On
Ubuntu/Debian:

```bash
apt update
apt install git-lfs
git lfs install
git lfs pull
```

The project uses Pixi environments. Common environments are:

- `sklearn-pypi`: vanilla scikit-learn benchmark runs
- `skl-cpu`: Array API CPU benchmark runs
- `skl-intel`: Intel Array API benchmark runs
- `skl-nvidia`: NVIDIA Array API benchmark runs
- `intel`: scikit-learn-intelex CPU and GPU benchmark runs
- `reporting`: dashboard generation and reporting utilities

## Architecture

The repository is split into a few layers:

- `configs/`: Python benchmark case generators. Public config scripts
  combine workload helpers with implementation selection and expose
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
orchestrator validates each case before running it. Per-workload case
generators live in `configs/synthetic_trees.py`, `configs/synthetic_linear.py`,
and `configs/real_datasets.py`; each exposes a `generate_cases(implem, tier)`
function that bakes a specific implementation dict into every case it yields.
Common utilities live in `configs/_common.py`, and Pixi-environment
implementation selection lives in `configs/_implementations.py`. The top-level
entry-point configs, `configs/all_models_test.py` and `configs/all_models_fast.py`,
loop over the implementations for the current Pixi environment, call each
workload generator per implementation, and merge the results - including
array-API implementations, which are filtered down to the estimators that
support them via `filter_array_api_supported_cases_if_needed`.

## Adding Benchmark Cases

Most case changes should start in `configs/synthetic_trees.py`,
`configs/synthetic_linear.py`, or `configs/real_datasets.py`, depending on the
workload.

Use `configs/all_models_test.py` for the current small exploratory matrix.
Use `configs/all_models_fast.py` when working on a broader but still
reasonably fast matrix. Both cover Array API Pixi environments as well as
plain sklearn/sklearnex ones.

Preview and validate a config by importing it directly:

```bash
pixi run -e sklearn-pypi python - <<'PY'
from sklbench.config import load_cases_from_script

cases = load_cases_from_script("configs/all_models_test.py")
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
pixi run -e sklearn-pypi python -m sklbench --config configs/all_models_test.py
```

`run.sh` runs the same `python -m sklbench` invocation across one or more
Pixi environments:

```bash
./run.sh env1 [env2 ...] [sklbench args...]
```

The leading arguments, up to the first one starting with `-`, are treated as
Pixi environments; everything from there on is passed through to `sklbench`
unchanged. For example, to reproduce the CPU benchmark set:

```bash
./run.sh sklearn-pypi --config configs/all_models_test.py
./run.sh skl-cpu --config configs/all_models_test.py
./run.sh intel --config configs/all_models_test.py
```

Add Intel GPU benchmarks via Array API (`./run.sh intel` above already covers
sklearnex GPU, filtered to the Ridge/LogisticRegression cases that support
it):

```bash
./run.sh skl-intel --config configs/all_models_test.py
```

Add NVIDIA GPU benchmarks:

```bash
./run.sh skl-nvidia --config configs/all_models_test.py
```

For ad hoc Intel or reporting work, prefer the explicit environments:

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
scikit-learn setup helper. It maintains a single scikit-learn checkout under
`sklearn-src/`, checks out the requested ref there, and installs that checkout
editable into `sklearn-dev`. `pixi.toml` points `sklearn-dev` at this exact
path.

```bash
scripts/setup_sklearn_ref.sh --ref main

pixi run -e sklearn-dev python -m sklbench --config configs/all_models_test.py
```

Run the same config against a branch from a fork by changing only the remote and
ref. Since the remote differs from the checkout's current origin, this recreates
the checkout (a fresh clone) rather than reusing it:

```bash
scripts/setup_sklearn_ref.sh \
  --remote https://github.com/some-user/scikit-learn.git \
  --ref my-perf-branch

pixi run -e sklearn-dev python -m sklbench --config configs/all_models_test.py
```

Pass orchestrator arguments directly to `sklbench` in the second command, for
example `--results-dir /tmp/sklbench-results` or `--exit-on-error`.

Only one scikit-learn ref is checked out at a time. Switching back to a
previous remote (e.g. back to upstream `main` after benchmarking a fork) also
recreates the checkout, so comparing two refs means re-running the setup
script between benchmark runs rather than keeping both checked out side by
side.

`run.sh` automates this re-running for you: give it `env@owner:ref` instead of
a plain environment name, and it runs `scripts/setup_sklearn_ref.sh` (against
`https://github.com/<owner>/scikit-learn.git`) before invoking `sklbench` for
that environment. This is the easiest way to compare a PR branch against a
base ref:

```bash
./run.sh sklearn-dev@cakedev0:hgb/use_threads_if sklearn-dev@scikit-learn:main \
    --config configs/hgb_scaling.py
```

Each `env@owner:ref` entry is set up and run in turn, so this checks out and
benchmarks the fork's branch, then re-checks-out and benchmarks upstream
`main`, both under the `sklearn-dev` Pixi environment. Reporting code tells
the two runs apart even though they share a Pixi environment name:
`sklbench.reporting.envs.software_build_name` labels a `sklearn-dev` build as
`sklearn-dev@<owner>:<short-commit>` whenever the environment is backed by a
separate scikit-learn git checkout, using the git commit metadata described
below (the checkout is always left in a detached-HEAD state, so the label
uses the commit rather than the branch name). That keeps
`sklearn-dev@cakedev0:hgb/use_threads_if` and `sklearn-dev@scikit-learn:main`
as distinct build variants in dashboards rather than merging them into one
`sklearn-dev` series.

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
  --ref bench/my-change
```

The software environment JSON records runtime import metadata for packages such
as `sklearn`, including git commit information when the imported package comes
from a git checkout.

## Previewing Dashboards Locally

Generate all dashboard pages into `_site/`:

```bash
for script in dashboards/gen_*.py; do
  pixi run -e reporting python "$script"
done
```

Open `_site/index.html` in a browser to inspect the local output.

During dashboard development, use the watcher to regenerate pages whenever
`results/`, `sklbench/reporting/`, or `dashboards/` changes:

```bash
pixi run -e reporting python watch_dashboards.py
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
