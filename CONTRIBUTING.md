# Contributing

This repository contains the benchmark configs, the benchmark results, and the
reporting scripts that generate the published scikit-learn benchmark
dashboards.

## Setup

Install pixi (>= 0.75):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Or update an older pixi:

```bash
pixi self-update
```

Install Git LFS before cloning:

```bash
pixi global install git-lfs
git lfs install
```

Then clone the repo. Result files are not downloaded by default (see
["Previewing Dashboards Locally"](#previewing-dashboards-locally)).

From the repo root, run:

```bash
./scripts/setup_sklearn_ref.sh --ref main
```

This checks out scikit-learn in `sklearn-src/`, which the `sklearn-dev`
environment needs (see ["Running Against scikit-learn
Branches"](#running-against-scikit-learn-branches)).

Then check your environment with:

```bash
pixi run -e sklearn-pypi python -m sklbench --config configs/smoke_check_test.py --results-dir ./results/tests/
```

It takes a few dozen seconds and writes a few results under `results/tests`
(ignored by git).

## Architecture

The project uses these Pixi environments:

- `sklearn-pypi`: vanilla scikit-learn
- `skl-cpu`: Array API on CPU (PyTorch CPU)
- `skl-intel`, `skl-nvidia`: Array API on Intel/NVIDIA GPUs (PyTorch, dpnp)
- `intel`: scikit-learn-intelex on CPU and GPU
- `reporting`: dashboard generation and reporting utilities
- `sklearn-cf-*`: conda-forge scikit-learn builds with different BLAS/OpenMP
  backends (see the `[environments]` table in `pixi.toml` for the exact
  matrix)
- `sklearn-dev`: scikit-learn built from a git checkout (see ["Running Against
  scikit-learn Branches"](#running-against-scikit-learn-branches))
- `sklearn-dev-libomp`: the same checkout as `sklearn-dev`, with the
  LLVM/Intel `libomp` runtime

The repository has a few layers:

- `configs/`: Python benchmark case generators. Public config scripts combine
  workload helpers with implementation selection and expose
  `generate_cases()`. See [configs/README.md](configs/README.md).
- `sklbench/`: the local benchmark package:
    - `sklbench/config/`: Pydantic case models and the config script loader.
    - `sklbench/orchestrator/`: captures the environment, launches one runner
      subprocess per case, captures logs and errors, and writes result files.
    - `sklbench/runners/`: runs a single expanded case; see
      [sklbench/runners/README.md](sklbench/runners/README.md).
    - `sklbench/reporting/`: result matching, environment summaries and HTML
      helpers. Apart from the initial result matching, it's fully vibe-coded.
- `results/`: benchmark outputs and environment metadata, tracked with Git
  LFS.
- `dashboards/index.py`: the only entry point for the full dashboard site. It
  generates the index page and calls `generate()` in each
  `dashboards/gen_*.py` module. Those modules read `results/` and write one
  HTML page each, and are not run directly. `dashboards/index_comparison.py`
  is the separate entry point for the ephemeral per-PR results of
  `pr-comparison.yml` (see COMPARISONS_PR.md). Fully vibe-coded.
- `.github/workflows/`: CI. `dashboard-pages.yml` runs `dashboards/index.py`
  on pushes to `main`. `dashboard-preview-build.yml` and
  `dashboard-preview-deploy.yml` build and deploy a preview dashboard for
  each results PR. `pr-comparison.yml` and `run-benchmarks.yml` run the
  benchmarks.

Config scripts are regular Python. `generate_cases()` returns a list of
JSON-serializable case dicts or pydantic case models, and the orchestrator
validates each case before running it. See
[configs/README.md](configs/README.md) for the per-workload generators and
tiers.

## Adding Benchmark Cases

Most case changes start in `configs/_synthetic_trees.py`,
`configs/_synthetic_linear.py` or `configs/_real_datasets.py`, depending on
the workload.

`configs/smoke_check_test.py` is a small matrix covering the Array API Pixi
environments as well as plain sklearn and sklearnex. CI's smoke check
workflows run it. No dashboard lists it in `SOURCE_CONFIGS` (see ["Config →
Dashboard Provenance"](#config--dashboard-provenance)): it checks the
orchestrator and config machinery, not the results.

Preview and validate a config by importing it:

```bash
pixi run -e sklearn-pypi python - <<'PY'
from sklbench.config import load_cases_from_script

cases = load_cases_from_script("configs/smoke_check_test.py")
print(len(cases))
print(cases[0])
PY
```

When adding cases, keep the matrix small enough to run repeatedly, set
estimator and data random states where relevant, and check that the
dashboard matching logic can compare each case to a baseline.

## Running Benchmarks

Run the default scikit-learn configuration:

```bash
pixi run -e sklearn-pypi python -m sklbench --config configs/smoke_check_test.py
```

`run.sh` runs the same `python -m sklbench` command in one or more Pixi
environments:

```bash
./run.sh env1 [env2 ...] [sklbench args...]
```

Arguments up to the first one starting with `-` are Pixi environments. The
rest is passed to `sklbench` unchanged. For example:

```bash
./run.sh sklearn-pypi sklearn-cf-mkl intel --config configs/all_models.py
```

Benchmark records are written under `results/`. To delete the results of a
test run, use `git clean results/ -fd` (this deletes all untracked results).

Once you have results locally, see ["Previewing Dashboards
Locally"](#previewing-dashboards-locally) to look at them.

## Running Against scikit-learn Branches

To check a performance PR locally, use one of the development Pixi
environments and the scikit-learn setup script. The script keeps a single
scikit-learn checkout in `sklearn-src/`, checks out the requested ref there,
and installs it in editable mode into `sklearn-dev`. `pixi.toml` points
`sklearn-dev` at this path.

```bash
scripts/setup_sklearn_ref.sh --ref main

pixi run -e sklearn-dev python -m sklbench --config configs/smoke_check_test.py
```

To run the same config on a branch from a fork, change the remote and ref.
Since the remote differs from the checkout's current origin, the script
clones it again instead of reusing the checkout:

```bash
scripts/setup_sklearn_ref.sh \
  --remote https://github.com/some-user/scikit-learn.git \
  --ref my-perf-branch

pixi run -e sklearn-dev python -m sklbench --config configs/smoke_check_test.py
```

Only one scikit-learn ref is checked out at a time. Switching back to a
previous remote (for example upstream `main` after a fork) also clones again.
To compare two refs, run the setup script again between the two benchmark
runs.

`run.sh` does this for you when given `env@owner:ref` entries, for example:

```bash
./run.sh sklearn-dev@cakedev0:hgb/use_threads_if sklearn-dev@scikit-learn:main \
    --config configs/hgb_scalability.py
```

Each `env@owner:ref` entry is set up and run in turn.

`sklearn-dev-libomp` installs the same `sklearn-src` checkout into a second
Pixi environment. To compare OpenMP runtimes on the same scikit-learn commit,
pass both environments to `run.sh` with the same ref:

```bash
./run.sh sklearn-dev@scikit-learn:main sklearn-dev-libomp@scikit-learn:main \
    --config configs/hgb_scalability.py
```

## PR Comparison Benchmarks

To benchmark an upstream scikit-learn PR or branch against `main` from a PR
on this repo, see [COMPARISONS_PR.md](COMPARISONS_PR.md).

## Previewing Dashboards Locally

By default, `results/` is checked out as Git LFS pointer files (small text
stubs), so a clone or `git pull` doesn't download the full results history.

Run the watcher. It downloads the missing result files of the current ref,
generates the pages, and generates them again whenever `results/`,
`sklbench/reporting/` or `dashboards/` changes:

```bash
pixi run -e reporting python watch_dashboards.py
```

To download all results of the current ref by hand, use
`git lfs pull --exclude ""` (a plain `git lfs pull` fetches nothing, see
["Notes"](#notes)).

## Config → Dashboard Provenance

Each dashboard declares which configs its data comes from:

- A runnable config is a `.py` file directly under `configs/` whose name
  doesn't start with an underscore (see `configs/README.md`). Leaf generators
  (`_real_datasets.py`, ...) and the `configs/_utils/` helpers start with an
  underscore so they don't look runnable.
- `sklbench.config.loader.load_cases_from_script` stamps every case with
  `metadata.source_config` (the repo-relative path of the config that was
  run). It is kept in `results/records/*.json`.
- Every `dashboards/gen_*.py` module declares `SOURCE_CONFIGS` (the configs
  it reads) and `SOURCE_ENVS` (the Pixi envs it expects) as module-level
  constants, and filters results with
  `sklbench.reporting.matching.matches_source_configs`.

To find what to rerun after a config change, use `scripts/what_to_rerun.py`:

```bash
pixi run -e reporting python scripts/what_to_rerun.py --dashboard all
pixi run -e reporting python scripts/what_to_rerun.py --dashboard "HistGradientBoosting thread-scalability breakdown" --hardware "Modern Intel laptop"
pixi run -e reporting python scripts/what_to_rerun.py --config configs/hgb_scalability.py
```

`--hardware` (a display name or hash from `dashboards.HARDWARE_NAMES`) keeps
only the envs that can be installed on that machine. This uses
`sklbench.reporting.envs.env_platforms()`, derived from `pixi.toml`, and
`dashboards.HARDWARE_PLATFORMS`.

Results older than this system (without `metadata.source_config`) match no
dashboard's `SOURCE_CONFIGS`, so they don't show up anywhere until rerun.

## Publishing New Results

Make sure Git LFS is set up (`git lfs install`, see ["Setup"](#setup)). You
don't need to pull existing results first: only the new files you add
matter.

Run the relevant benchmarks, then check the generated files:

```bash
git status --short results/
```

Stage the new result and environment JSON files together:

```bash
git add results/
```

Commit and push on a branch and open a PR (for example "RES HGB Macbook
results"). The PR deploys a preview dashboard with your results, as long as a
dashboard's `SOURCE_CONFIGS` lists the config you ran (see ["Config →
Dashboard Provenance"](#config--dashboard-provenance)).

A results PR should only contain results. Open a separate PR for fixes,
enhancements, new configs, etc.

Once the PR is merged into `main`, the GitHub Pages workflow regenerates and
deploys the dashboards.

## Notes

`results/*.json` and `results/**/*.json` are tracked with Git LFS through
`.gitattributes`. Don't bypass LFS for benchmark results.

`.lfsconfig` sets `fetchexclude = *`, so clones and plain `git lfs
pull`/`fetch` don't download `results/` content by default. This protects the
repo's LFS quota (see ["Previewing Dashboards
Locally"](#previewing-dashboards-locally)). CI workflows pass their own
`--include`, but `-I` alone only overrides `fetchinclude`: `fetchexclude = *`
still applies and blocks everything, so these workflows must also pass
`--exclude ""`.

**Cloud machines can have high tail variability**, especially for scaling
studies and short workloads. Prefer stable local or dedicated hardware when
deciding whether one representative case can replace a broader matrix.
