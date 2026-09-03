import configs.hgb_scalability as hgb_scalability


# Real-dataset cases don't carry n_samples/n_features in the case dict (see
# hgb_scalability.py's _real_dataset_cases), so "small" for those is decided
# via this whitelist instead of a size computation. Sizes are documented in
# sklbench/runners/datasets/loaders.py: ames_housing is 1460 x 79 (~115K),
# amazon_employee_access is 32769 x 9 (~295K). The other datasets in
# hgb_scalability.REAL_SCALING_DATASETS (kddcup09_churn, year_prediction_msd,
# covtype, susy) are all well past the 1M nxd threshold.
SMALL_REAL_DATASETS = {"ames_housing", "amazon_employee_access"}

MAX_N_SAMPLES_X_N_FEATURES = 1_000_000


def _is_single_threaded(case: dict) -> bool:
    return case["bench"]["env"]["OMP_NUM_THREADS"] == "1"


def _is_small(case: dict) -> bool:
    generation_kwargs = case["data"].get("generation_kwargs")
    if generation_kwargs is not None:
        n_samples_x_n_features = (
            generation_kwargs["n_samples"] * generation_kwargs["n_features"]
        )
        return n_samples_x_n_features <= MAX_N_SAMPLES_X_N_FEATURES
    return case["data"]["dataset"] in SMALL_REAL_DATASETS


def generate_cases() -> list[dict]:
    return [
        case
        for case in hgb_scalability.generate_cases()
        if _is_single_threaded(case) and _is_small(case)
    ]
