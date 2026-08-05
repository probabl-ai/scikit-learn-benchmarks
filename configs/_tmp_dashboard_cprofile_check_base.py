def generate_cases() -> list[dict]:
    return [
        {
            "algorithm": {"estimator": "Ridge"},
            "data": {
                "source": "make_regression",
                "generation_kwargs": {"n_samples": 1000, "n_features": 20, "n_informative": 10},
                "order": "C",
                "dtype": "float64",
            },
            "implementation": {"library": "sklearn"},
            "bench": {"n_runs": 3, "time_limit": 4, "py_spy_profiling": True},
        }
    ]
