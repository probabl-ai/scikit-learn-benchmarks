"""
Regression counterpart of `gbdt_pareto.py`: the same ladder of settings,
scored by R2.
"""
import configs.gbdt_pareto as gbdt_pareto

DATASETS = [
    # 1.5k x 79, mostly categorical: the smallest dataset.
    "ames_housing",
    # 21k x 8, numeric only.
    "california_housing",
    # 163k x 11, categoricals with ~2000-3300 categories.
    "medical_charges_nominal",
    # 515k x 90, numeric only.
    "year_prediction_msd",
]


def generate_cases() -> list[dict]:
    return gbdt_pareto.generate_task_cases(DATASETS, "regression")
