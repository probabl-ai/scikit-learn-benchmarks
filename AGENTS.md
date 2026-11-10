# Repository Notes

- Check `CONTRIBUTING.md` for repository workflows before changing benchmark
  configs, results, dashboard generation, or publishing-related code.
- Use the `intel` pixi environment for Intel and array API benchmark work:
  `pixi run -e intel ...`.
- Use the `reporting` pixi environment for dashboard and reporting work:
  `pixi run -e reporting ...`.
- Run all Python commands through Pixi using the appropriate repository
  environment; do not invoke the system `python`, `python3`, or environment
  binaries directly unless Pixi itself is unavailable.
- During dashboard iteration, assume the user is running
  `pixi run -e reporting python watch_dashboards.py` on the side to regenerate
  dashboard HTML when `results/`, `sklbench/reporting/`, or `dashboards/` changes. Do not
  start a second watcher unless explicitly asked; use one-shot generator runs or
  `py_compile` only when a validation check is useful.
- `dashboards/index.py` is the sole entry point for the full dashboard site
  (pushes to `main`, PR previews, local `watch_dashboards.py`);
  `dashboards/index_comparison.py` is the separate entry point for PR
  comparison runs (see COMPARISONS_PR.md). Each other `dashboards/gen_*.py`
  module reads benchmark data from `results/` and exposes a
  `generate(output_dir)` function that these two call - it is not run
  directly and takes no `--output-dir` of its own.
- `sklbench/` is the local benchmark package. Changes under `sklbench/` belong
  to this repository.
- Do not spend time preserving or normalizing LF vs CRLF line endings in this
  repository. Mixed line endings are acceptable here unless a tool actually
  fails because of them.
- Prefer validating config changes by importing the config script and calling
  `generate_cases()` before running full benchmarks.
- Do not add tests for small UX/log-noise details, such as filtering known
  benign command-line warnings from orchestrator output.

# Experimentation guidelines

These apply to ad hoc timing experiments on the benchmark machines (A/B tests,
library comparisons, reruns of an upstream PR's benchmark), not only to the
`sklbench` runs.

- Time everything under `benchlock`, which is on the PATH of all benchmark
  machines (`benchlock --help`). Check `benchlock status --all-hosts` first,
  always pass `--tag` with a short description of the task, and also take the
  lock for CPU-heavy setup (`pixi install`, builds, data preparation). If it
  gives up waiting (exit 75), pick another machine or slot instead of retrying
  in a loop.
- Run all the contenders of a comparison interleaved inside one `benchlock`
  call, not one call per contender.
- Launch long runs detached so they survive the ssh session:
  `ssh -f HOST 'cd DIR && nohup benchlock --tag "..." all -- ./run.sh > run.log 2>&1 < /dev/null &'`.
  To stop one, kill the run script and its Python processes as well as
  `benchlock`; killing only `benchlock` leaves the tree running.
- Append one JSON line per measurement as soon as it finishes, with the
  library versions, so a stopped or crashed run keeps what it measured.
- Keep one-time costs (imports, data loading, compilation, warm-up) out of the
  timed region, and set the same thread counts and precision settings for all
  contenders.
- Check that the contenders do the same work: log the number of iterations and
  a quality metric next to each time. A faster run that stops earlier is not
  faster.
- Estimate the total duration from the first measured fits before announcing
  it, and ask the user before starting anything longer than ~1 hour.
- If ssh reports a changed host key, stop and ask the user before accepting
  the new key.
