import hgb_scaling


def generate_cases() -> list[dict]:
    return [
        {
            **case,
            "bench": {
                **case["bench"],
                "env": {
                    **case["bench"]["env"],
                    "GOMP_SPINCOUNT": "300000",
                },
            },
        }
        for case in hgb_scaling.generate_cases()
    ]
