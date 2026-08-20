import configs.hgb_scaling_taskset as hgb_scaling_taskset


def generate_cases() -> list[dict]:
    return [
        {
            **case,
            "bench": {
                "n_runs": 3,
                "env": {
                    "OMP_PROC_BIND": True,
                    "OMP_NUM_THREADS": case["bench"]["env"]["OMP_NUM_THREADS"],
                },
                "py_spy_profiling": False,
                "taskset": case["bench"]["taskset"]
            },
        }
        for case in hgb_scaling_taskset.generate_cases()
    ]
