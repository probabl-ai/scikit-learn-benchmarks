# Repository Notes

- Check `CONTRIBUTING.md` for repository workflows before changing benchmark
  configs, results, dashboard generation, or publishing-related code.
- Use the `intel` pixi environment for Intel and array API benchmark work:
  `pixi run -e intel ...`.
- Use the `reporting` pixi environment for dashboard and reporting work:
  `pixi run -e reporting ...`.
- During dashboard iteration, assume the user is running
  `pixi run -e reporting python watch_dashboards.py` on the side to regenerate
  dashboard HTML when `results/`, `reporting/`, or `dashboard/` changes. Do not
  start a second watcher unless explicitly asked; use one-shot generator runs or
  `py_compile` only when a validation check is useful.
- Dashboard entry points live in `dashboard/gen_*.py`, read benchmark data from
  `results/`, and should write HTML through `dashboard/output.py` so
  `--output-dir` works with the watcher and publishing workflow.
- `sklbench` is a symlink to the nested `scikit-learn_bench/sklbench`
  checkout. Changes under `sklbench/` belong to that nested repository.
- Do not spend time preserving or normalizing LF vs CRLF line endings in this
  repository. Mixed line endings are acceptable here unless a tool actually
  fails because of them.
- Prefer validating config changes with `validate_config.py` and previewing
  expansions with `preview_cases.py` before running full benchmarks.
