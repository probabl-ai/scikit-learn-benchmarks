"""
Benchmark for OneHotEncoder / OrdinalEncoder / TargetEncoder `fit`/`transform`
performance on the categorical columns of a few real datasets, across the
input dtypes those encoders commonly see in the wild (`object`, pandas
`string`, `category`).

Adapted from scikit-learn PR #34394
(https://github.com/scikit-learn/scikit-learn/pull/34394, "BENCH Add
benchmark for encoders"): same encoders/datasets/dtype axis, but running
through this repo's config/orchestrator/runner machinery instead of a
standalone script. Left out for now, relative to that PR:

- the pandas-vs-polars axis (would be `implementation.data_library`) - this
  repo's data-conversion layer
  (`sklbench.runners.datasets.transformer.convert_data`) has no polars
  support yet;
- its `--with-numericals` mode (encoder wrapped in a ColumnTransformer with
  numeric passthrough) - doesn't fit the plain
  `estimator_class(**estimator_params)` case model without a dedicated
  mechanism (columns to pass through differ per dataset).

Every dataset below (see `sklbench/runners/datasets/loaders.py`) already
stores its categorical columns as pandas `category` dtype.
`data:preprocessing_kind="categorical_only"`
(`sklbench/runners/datasets/preprocessing.py`) drops the numeric columns and
casts the remaining categorical columns to `data:preprocessing_kwargs:format`
- not the generic `data:dtype` knob, which would also cast `y` and break
`TargetEncoder`'s target-type detection on regression datasets (no
`n_classes` to force it back to numeric).

Note: `EstimatorCase.name()` doesn't key off `preprocessing_kwargs`, so the
three format variants of a given (encoder, dataset) pair share the same case
name/label - harmless for the JSONL results themselves (each case file is
keyed by a hash of its full content, not its name) but worth knowing if a
dashboard is later built on top of these results.
"""

DATASETS = [
    "ames_housing",  # regression, 1460 rows, mostly low-cardinality
    "amazon_employee_access",  # classification, 32769 rows, high-cardinality
    "kddcup09_churn",  # classification, 50000 rows, heavy missingness
    "kick",  # classification, 72983 rows, wide cardinality range
    "bank_marketing",  # classification, 45211 rows, low-cardinality
]

FORMATS = ["object", "string", "category"]

# TargetEncoder's `target_type="auto"` detects continuous/binary/multiclass,
# so the same params work across both the regression and classification
# datasets above. To try a different variant of one of these encoders (e.g.
# a smaller `max_categories`), edit its entry here directly - adding a
# second variant of the same encoder class instead would collide with the
# first in case naming (see the module docstring).
ENCODER_PARAMS = {
    "OneHotEncoder": {"handle_unknown": "ignore", "max_categories": 20},
    "OrdinalEncoder": {
        "handle_unknown": "use_encoded_value",
        "unknown_value": -2,
        "encoded_missing_value": -1,
    },
    "TargetEncoder": {"target_type": "auto"},
}


def generate_cases() -> list[dict]:
    cases = []
    for dataset in DATASETS:
        for dtype in FORMATS:
            for estimator, estimator_params in ENCODER_PARAMS.items():
                cases.append(
                    {
                        "implementation": {"library": "sklearn"},
                        "algorithm": {
                            "estimator": estimator,
                            "estimator_params": estimator_params,
                        },
                        "data": {
                            "dataset": dataset,
                            "preprocessing_kind": "categorical_only",
                            "preprocessing_kwargs": {"format": dtype},
                        },
                        "metadata": {"format": dtype},
                    }
                )
    return cases
