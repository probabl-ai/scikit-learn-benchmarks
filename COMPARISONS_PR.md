# PR Comparison Benchmarks

A pull request on this repo can automatically benchmark an upstream
scikit-learn PR against `main` and post a before/after comparison dashboard
back on the PR. To trigger it, put a fenced `sklbench-compare` block
somewhere in the PR description:

```sklbench-compare
sklearn_ref: cakedev0:ridge/optim_cholesky
runs: configs/hgb_scaling.py, intel-gnr#sklearn-dev-libomp#configs/pipeline.py
```

- `sklearn_ref` (required): the scikit-learn fork owner and branch/ref to
  compare against `main`, as `owner:ref` - the same shorthand `run.sh`'s
  `env@owner:ref` spec takes (see CONTRIBUTING.md's "Running Against
  scikit-learn Branches").
- `runs` (required): a comma/whitespace-separated list of
  `[runner#][env#]config` entries - one explicit (runner, env, config) tuple
  per entry, **not** a cross product of separate runner/env/config lists.
  Each entry ends in exactly one `configs/<name>.py` path; before it, `runner`
  and `env` are both optional and can appear in either order (their allowed
  values don't overlap, so a bare token is recognized by what it is, not by
  position):
  - `runner`: `intel-laptop`, `intel-gnr`, or `both` - which self-hosted
    machine(s) run this entry. Defaults to `both` when omitted.
  - `env`: a pixi env that path-depends on `sklearn-src` (see pixi.toml) -
    currently `sklearn-dev` or `sklearn-dev-libomp` - which pixi env builds
    both sides of the `sklearn_ref` vs `main` comparison for this entry.
    Defaults to `sklearn-dev` when omitted.

  So `configs/hgb_scaling.py` alone runs that config on both runners under
  `sklearn-dev` (the original, single-config/single-env behavior).
  `intel-gnr#sklearn-dev-libomp#configs/pipeline.py` runs only on `intel-gnr`,
  building both `main` and the PR ref under `sklearn-dev-libomp`, to isolate
  e.g. an OpenMP-runtime effect the way CONTRIBUTING.md's `run.sh` example
  does manually. A directive can mix any number of such entries; each
  self-hosted runner referenced by at least one entry gets its own CI job,
  and every entry targeting that runner accumulates into that job's one
  results dir before it generates its one dashboard - so results from the
  same machine always land in the same dashboard, with an env/branch filter
  column to tell entries apart when more than one env is involved.

Only PRs opened by someone with write access to this repo (`OWNER`/`MEMBER`/
`COLLABORATOR`) trigger the self-hosted benchmark run, and **the PR branch
must be pushed directly to `scikit-learn-benchmarks`, not a personal fork**:
GitHub withholds secrets and a write-scoped token from `pull_request` runs
whenever the PR's head repo differs from the base repo, regardless of
permissions, so a fork-sourced PR can run the benchmark but can't deploy the
dashboard or comment on the PR. Results are ephemeral (never committed to
`results/`); each new push re-triggers the comparison. See
`.github/workflows/pr-comparison.yml`.
