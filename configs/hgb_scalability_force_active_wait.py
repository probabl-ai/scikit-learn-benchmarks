import configs.hgb_scalability as hgb_scalability


def generate_cases() -> list[dict]:
    return [
        {
            **case,
            "bench": {
                "n_runs": 5,
                "env": {
                    "GOMP_SPINCOUNT": "300000",
                    "KMP_BLOCKTIME": "200ms",
                    "OMP_NUM_THREADS": case["bench"]["env"]["OMP_NUM_THREADS"],
                },
                "py_spy_profiling": False,
            },
        }
        for case in hgb_scalability.generate_cases()
    ]
