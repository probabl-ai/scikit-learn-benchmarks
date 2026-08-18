import hgb_scaling


def generate_cases() -> list[dict]:
    return [
        {
            **case,
            "bench": {
                "n_runs": 5,
                "env": {
                    # "GOMP_SPINCOUNT": "300000",
                    "OMP_NUM_THREADS": case["bench"]["env"]["OMP_NUM_THREADS"],
                },
                "py_spy_profiling": False,
            },
        }
        for case in hgb_scaling.generate_cases()
    ]
