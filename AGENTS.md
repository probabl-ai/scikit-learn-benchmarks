# Repository Notes

- Check `CONTRIBUTING.md` for repository workflows before changing benchmark
  configs, results, dashboard generation, or publishing-related code.
- Use the `intel` pixi environment for Intel and array API benchmark work:
  `pixi run -e intel ...`.
- Use the `reporting` pixi environment for dashboard and reporting work:
  `pixi run -e reporting ...`.
- During dashboard iteration, assume the user is running
  `pixi run -e reporting python watch_dashboards.py` on the side to regenerate
  dashboard HTML when `results/`, `sklbench/reporting/`, or `dashboards/` changes. Do not
  start a second watcher unless explicitly asked; use one-shot generator runs or
  `py_compile` only when a validation check is useful.
- Dashboard entry points live in `dashboards/gen_*.py`, read benchmark data from
  `results/`, and should write HTML through `dashboards/output.py` so
  `--output-dir` works with the watcher and publishing workflow.
- `sklbench/` is the local benchmark package. Changes under `sklbench/` belong
  to this repository.
- Do not spend time preserving or normalizing LF vs CRLF line endings in this
  repository. Mixed line endings are acceptable here unless a tool actually
  fails because of them.
- Prefer validating config changes by importing the config script and calling
  `generate_cases()` before running full benchmarks.
- Do not add tests for small UX/log-noise details, such as filtering known
  benign command-line warnings from orchestrator output.
