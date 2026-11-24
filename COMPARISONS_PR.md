# PR Comparison Benchmarks

A pull request on this repo can benchmark an upstream scikit-learn PR against
`main` and post a before/after comparison dashboard on the PR. To trigger it,
put a fenced `sklbench-compare` block in the PR description:

```sklbench-compare
sklearn_ref: cakedev0:ridge/optim_cholesky
runs: configs/hgb_scalability.py, intel-xeon#sklearn-dev-libomp#configs/pipeline.py
```

- `sklearn_ref` (required): the scikit-learn fork owner and branch or ref to
  compare against `main`, as `owner:ref`. This is the same shorthand as
  `run.sh`'s `env@owner:ref` (see "Running Against scikit-learn Branches" in
  CONTRIBUTING.md).
- `sklearn_base_ref` (optional): the ref to compare `sklearn_ref` against, in
  the same `owner:ref` form. Defaults to `scikit-learn:main`. Both refs can be
  full commit shas, for example to measure a merged PR in isolation: its merge
  commit as `sklearn_ref` and the parent of that commit as `sklearn_base_ref`.
  With a base ref, read `main` below as that ref (`[skip main]` reuses its
  results).
- `runs` (required): a comma or whitespace separated list of
  `[runner#][env#]config` entries. Each entry is one explicit (runner, env,
  config) tuple, **not** a cross product of runner, env and config lists.
  Each entry ends with exactly one `configs/<name>.py` path. Before it,
  `runner` and `env` are both optional and can come in any order (their
  values don't overlap, so each token is recognized by its value):
  - `runner`: `intel-laptop`, `intel-xeon`, `intel-all`, `aws-nvidia-l4` or
    `all`, the self-hosted machine(s) that run this entry. Defaults to
    `intel-all`, the two Intel machines: `intel-laptop` and `intel-xeon`.
    `all` is every machine: the two Intel machines plus `aws-nvidia-l4`.
  - `env`: a pixi env that depends on `sklearn-src` (see `pixi.toml`),
    currently `sklearn-dev`, `sklearn-dev-libomp`, `sklearn-dev-mkl`,
    `sklearn-dev-freethreading` or `sklearn-dev-nvidia`, or `auto`. Defaults
    to `auto`, the environment each machine is meant to build with:
    `sklearn-dev` on `intel-laptop` and `intel-xeon`, `sklearn-dev-nvidia`
    (the CUDA-capable one) on `aws-nvidia-l4`. Writing `auto` explicitly
    means exactly the same as leaving it out. A concrete env wins over
    `auto`, and builds both sides of the `sklearn_ref` vs `main` comparison
    for this entry. A machine with no `auto` mapping is a hard error naming
    the machine.

  For example, `configs/hgb_scalability.py` alone runs that config on both
  Intel hosts, each with the env it is meant to use, `sklearn-dev`.
  `all#configs/hgb_scalability.py` covers the L4 too, under
  `sklearn-dev-nvidia`. `intel-xeon#sklearn-dev-libomp#configs/pipeline.py`
  runs only on `intel-xeon` and builds both `main` and the PR ref with
  `sklearn-dev-libomp`, to isolate an OpenMP runtime effect like the `run.sh`
  example in CONTRIBUTING.md.

  A block can mix any number of entries. Each runner used by at least one
  entry gets its own CI job, which accumulates the results of all its
  entries and generates one dashboard. Results from the same machine always
  land in the same dashboard, with an env/branch column to tell entries apart
  when several envs are involved.

  **A bare entry runs on the two Intel machines and starts no GPU host.** A
  bare entry covers `intel-laptop` and `intel-xeon`, and `auto` gives both
  `sklearn-dev`. The workflow runs no provision, runner-ready or cleanup job
  for it and leaves the L4 instance down. A comparison that includes the GPU
  host names it in the entry: `all` for every machine, e.g.
  `all#configs/x.py`, or `aws-nvidia-l4` for the GPU host alone, e.g.
  `aws-nvidia-l4#sklearn-dev-nvidia#configs/all_models.py`. A GPU entry
  starts the instance on demand, waits for its runner to come online, then
  leaves it running for the next comparison.

The before/after table is always generated. To also publish one of the other
`dashboards/gen_*.py` dashboards on a run's results, call it from the
`__main__` of `dashboards/index_comparison.py` and include that edit in your
PR. This is useful when a specialized dashboard reads a config's results
better than the generic table.

### Skipping the `main` re-benchmark

By default, every push benchmarks scikit-learn `main` again next to the PR
ref, even though `main` usually hasn't changed between two pushes. Add
`[skip main]` to the latest commit message to reuse the `main` results this
runner last published instead, downloaded from its deployed site. The PR ref
is always benchmarked again.

If no cached results are available yet (for example on the first comparison
of a PR, or when the previous `main` run didn't complete), `main` is
benchmarked anyway and the run logs a warning. Each runner's PR comment says
whether its `main` results were reused or benchmarked again.

### Rebuilding only the dashboard

Add `[skip]` to the latest commit message to run no benchmark at all. Each
runner's job downloads the result files its comparison last published (both
`main` and the PR ref), regenerates the dashboard with this commit's code and
redeploys it. Use it when a push only changes dashboard code, for example
`dashboards/index_comparison.py` or `sklbench/reporting/`.

These jobs run on `ubuntu-latest` instead of the self-hosted runners, so they
don't wait for a benchmark to free the runner. The `sklbench-compare` block
must still be present and valid, since it selects which runners' sites to
rebuild, but its `sklearn_ref` and configs are not run. With the default
runner that means both Intel runners' sites, built on `ubuntu-latest`: the
jobs contact no self-hosted machine and start no L4 host. If a runner has no
published results yet, its job fails: push without `[skip]` first, or narrow
the directive to the runners that do have results (e.g. `intel-xeon#...`).

A benchmark run publishes its results before it generates the dashboard, so
if the dashboard step fails, fix it and push a `[skip]` commit: the results
it already published are reused.

### Permissions

Only PRs opened by someone with write access to this repo
(`OWNER`/`MEMBER`/`COLLABORATOR`) trigger the self-hosted benchmark run, and
**the PR branch must be pushed to `scikit-learn-benchmarks` itself, not to a
personal fork**. GitHub withholds secrets and a write-scoped token from
`pull_request` runs when the head repo differs from the base repo, whatever
the permissions. A PR from a fork can run the benchmark, but it can't deploy
the dashboard or comment on the PR.

Results are ephemeral: they are never committed to `results/`, and each new
push triggers the comparison again. See `.github/workflows/pr-comparison.yml`.
