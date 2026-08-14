import hgb_scaling_dev

KEPT_WORKLOAD_NAMES = {"L"}
KEPT_DATASETS = {"kddcup09_churn"}


def generate_cases() -> list[dict]:
    return [
        case
        for case in hgb_scaling_dev.generate_cases()
        if case["metadata"].get("name") in KEPT_WORKLOAD_NAMES
        or case["data"].get("dataset") in KEPT_DATASETS
    ]
