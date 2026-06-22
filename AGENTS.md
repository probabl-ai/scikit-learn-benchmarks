# Repository Notes

- Use the `intel` pixi environment for Intel and array API benchmark work:
  `pixi run -e intel ...`.
- Use the `reporting` pixi environment for dashboard and reporting work:
  `pixi run -e reporting ...`.
- `sklbench` is a symlink to the nested `scikit-learn_bench/sklbench`
  checkout. Changes under `sklbench/` belong to that nested repository.
- Do not spend time preserving or normalizing LF vs CRLF line endings in this
  repository. Mixed line endings are acceptable here unless a tool actually
  fails because of them.
- Prefer validating config changes with `validate_config.py` and previewing
  expansions with `preview_cases.py` before running full benchmarks.
