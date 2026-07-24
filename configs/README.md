# Benchmark Config Scripts

Each Python file in this directory defines a benchmark matrix through a
`generate_cases()` function. The function must return a list of plain
dictionaries or pydantic case models exported by `sklbench.config`.

Shared workload helpers live in `configs/_generators.py`; implementation
selection lives in `configs/_implementations.py`.

Public estimator configs are split by workload and size:

- `all_models_test.py` and `all_models_fast.py` for the general model matrix.
- `array_api_test.py` and `array_api_fast.py` for Array API-compatible cases.

Run them through Pixi so `PIXI_ENVIRONMENT_NAME` selects the implementation set.
