# Benchmark Config Scripts

Each Python file in this directory defines a benchmark matrix through a
`generate_cases()` function. The function must return a list of plain
dictionaries or pydantic case models exported by `sklbench.config`.

Per-workload case generators live in `configs/synthetic_trees.py`,
`configs/synthetic_linear.py`, and `configs/real_datasets.py`. Each exposes
`generate_cases(implem, tier)`, taking a single implementation dict and
baking it directly into every case it yields (no separate cross-product step
needed downstream). `tier`/`max_tier` is one of `"test"`, `"fast"`,
`"normal"`, or `"slow"`; `"normal"`/`"slow"` in the synthetic generators chain
several increasing `scale` values (mirroring `synthetic_trees.py`'s ladder)
and deterministically subsample the resulting matrix down to about a third,
so the case count stays manageable as more scales are added. Common
utilities live in `configs/_common.py`; implementation selection lives in
`configs/_implementations.py`.

Public estimator configs:

- `all_models_test.py`, `all_models_fast.py`, and `all_models.py` for the
  `"test"`, `"fast"`, and `"normal"` tiers of the general model matrix,
  spanning plain sklearn/sklearnex Pixi environments as well as Array API
  ones. Array API cases are filtered down to the estimators that support them
  via `sklbench.config.utils.filter_array_api_supported_cases_if_needed`.

Run them through Pixi so `PIXI_ENVIRONMENT_NAME` selects the implementation set.
