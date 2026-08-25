# PR Comparison Benchmarks

A pull request on this repo can automatically benchmark an upstream
scikit-learn PR against `main` and post a before/after comparison dashboard
back on the PR. To trigger it, put a fenced `sklbench-compare` block
somewhere in the PR description:

```sklbench-compare
sklearn_ref: cakedev0:ridge/optim_cholesky
config: configs/hgb_scaling.py
runners: intel-laptop, intel-gnr
```

- `sklearn_ref` (required): the scikit-learn fork owner and branch/ref to
  compare against `main`, as `owner:ref` - the same shorthand `run.sh`'s
  `env@owner:ref` spec takes (see CONTRIBUTING.md's "Running Against
  scikit-learn Branches").
- `config` (required): a single `configs/<name>.py` path (one config per run;
  not yet supported: multiple configs in one directive).
- `runners` (optional, default both): `intel-laptop`, `intel-gnr`, or `both`.

Only PRs opened by someone with write access to this repo (`OWNER`/`MEMBER`/
`COLLABORATOR`) trigger the self-hosted benchmark run, and **the PR branch
must be pushed directly to `scikit-learn-benchmarks`, not a personal fork**:
GitHub withholds secrets and a write-scoped token from `pull_request` runs
whenever the PR's head repo differs from the base repo, regardless of
permissions, so a fork-sourced PR can run the benchmark but can't deploy the
dashboard or comment on the PR. Results are ephemeral (never committed to
`results/`); each new push re-triggers the comparison. See
`.github/workflows/pr-comparison.yml`.
