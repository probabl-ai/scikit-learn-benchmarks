# Contributing

This repository contains benchmark configurations, captured benchmark results,
and reporting scripts for the published scikit-learn benchmark dashboards.

## Setup


### Pre-requisites

Install pixi (>= 0.75): 

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Or if you have an old pixi, you can self-update:
```bash
pixi self-update 
```

Install Git LFS before cloning:
```
pixi global install git-lfs
git lfs install
```

Then clone the repo (`git lfs pull` afterwards if you want to fetch all results locally).


Then from the repo root run:

```bash
./scripts/setup_sklearn_ref.sh --ref main
```

The `sklearn-dev` environment (see ["Running Against scikit-learn
Branches"](#running-against-scikit-learn-branches) below) depends on scikit-learn
being checked out in a local path (`sklearn-src/`); running this script installs it.

Then you test your environnment by running:

```bash
pixi run -e sklearn-pypi python -m sklbench --config configs/all_models_test.py --results-dir ./results/tests/
```

It will take a few dozen seconds and create a few results under `results/tests` (git ignored).


## Architecture

The project uses Pixi environments:
- `sklearn-pypi`: vanilla scikit-learn
- `skl-cpu`: Array API CPU (pytorch CPU)
- `skl-intel`, `skl-nvidia`: Intel/NVIDIA GPU Array API (pytorch, dpnp)
- `intel`: scikit-learn-intelex CPU and GPU
- `reporting`: dashboard generation and reporting utilities
- `sklearn-cf-*`: conda-forge scikit-learn builds across BLAS/OpenMP backends
  (see the `[environments]` table in `pixi.toml` for the exact matrix)
- `sklearn-dev`: scikit-learn built from a git checkout (see ["Running Against
  scikit-learn Branches"](#running-against-scikit-learn-branches) below)
- `sklearn-dev-libomp`: same scikit-learn git checkout as `sklearn-dev`, but
  with LLVM/Intel `libomp` runtime.


The repository is split into a few layers:

- `configs/`: Python benchmark case generators. Public config scripts
  combine workload helpers with implementation selection and expose
  `generate_cases()`. See [configs/README.md](configs/README.md).
- `sklbench/`: local benchmark package containing:
    - `sklbench/config/`: Pydantic case models and the config-script loader.
    - `sklbench/orchestrator/`: captures the environment, launches one runner
      subprocess per case, captures logs/errors, and writes result files.
    - `sklbench/runners/`: executes a single already-expanded case; see
      [sklbench/runners/README.md](sklbench/runners/README.md).
    - `sklbench/reporting/`: result matching, environment summaries, and HTML
      helpers. Except the initial results-matching work, it's fully vibe-coded.
- `results/`: captured benchmark outputs and environment metadata, tracked
  with Git LFS.
- `dashboards/gen_*.py`: dashboard entry points. Each script reads `results/`
  and writes one HTML page. Fully vibe-coded.
- `.github/workflows/`: CI. `dashboard-pages.yml` runs all `dashboards/gen_*.py`
  on pushes to `main`; `dashboard-preview-build.yml`/`dashboard-preview-deploy.yml`
  build and deploy a preview dashboard per results PR; `pr-comparison.yml` and
  `run-benchmarks.yml` drive the benchmark-running workflows.

Config scripts are regular Python: they return a list of JSON-serializable case
dictionaries or pydantic case models from `generate_cases()`, and the
orchestrator validates each case before running it. See
[configs/README.md](configs/README.md) for the per-workload generator layout
and tier conventions.

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
unchanged. For example:

```bash
./run.sh sklearn-pypi sklearn-cf-mkl intel --config configs/all_models.py
```

Generated benchmark records are written under `results/`. If you need to clean
results from a test/rehearsal run, you can do: `git clean results/ -fd`
(deletes all untracked results).

Once you have some results locally, see ["Previewing Dashboards
Locally"](#previewing-dashboards-locally) below to check them.

## Running Against scikit-learn Branches

For local performance PR checks, use one of the development Pixi environments and
the scikit-learn setup helper. It maintains a single scikit-learn checkout under
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

Only one scikit-learn ref is checked out at a time. Switching back to a
previous remote (e.g. back to upstream `main` after benchmarking a fork) also
recreates the checkout, so comparing two refs means re-running the setup
script between benchmark runs rather than keeping both checked out side by
side.

`run.sh` automates this re-running for you: give it `env@owner:ref`, for example:

```bash
./run.sh sklearn-dev@cakedev0:hgb/use_threads_if sklearn-dev@scikit-learn:main \
    --config configs/hgb_scalability.py
```

Each `env@owner:ref` entry is set up and run in turn.

Note: `sklearn-dev-libomp` installs the same `sklearn-src` checkout into a second
Pixi environment. Use it to compare OpenMP runtimes on the exact same scikit-learn commit,
by passing both environments to `run.sh` with the same ref:

```bash
./run.sh sklearn-dev@scikit-learn:main sklearn-dev-libomp@scikit-learn:main \
    --config configs/hgb_scalability.py
```

## PR Comparison Benchmarks

Automatically benchmarking an upstream scikit-learn PR/branch against `main` from a
PR description on this repo is documented separately, in [COMPARISONS_PR.md](COMPARISONS_PR.md).

## Previewing Dashboards Locally

During dashboard development, use the watcher to regenerate pages whenever
`results/`, `sklbench/reporting/`, or `dashboards/` changes:

```bash
pixi run -e reporting python watch_dashboards.py
```

## Publishing New Results

Before committing results, make sure lfs is set-up:

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

Commit and push on a branch and open a PR (e.g. "RES HGB Macbook results"); it
will deploy a preview dashboard including your results (if they match the
filters of some dashboard).

Make sure a results PR only contains results - open a separate PR for
fixes/enhancements/new configs/etc.

Once the change reaches `main`, the GitHub Pages workflow regenerates and deploys the
dashboards automatically.

## Notes

`results/*.json` and `results/**/*.json` are tracked through Git LFS via
`.gitattributes`. Do not bypass LFS for benchmark results.

**Cloud machines can have high tail variability**, especially for scaling studies
and short workloads. Prefer stable local/dedicated hardware when deciding whether
one representative case can replace a broader matrix.
