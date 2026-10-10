"""
`gbdt_pareto.py` with OpenMP active wait forced, to measure how much each
library depends on the OpenMP runtime's wait policy (conda-forge's libgomp
defaults to passive wait). CatBoost doesn't use OpenMP, so it's left out.
"""
import configs.gbdt_pareto as gbdt_pareto


def with_active_wait(cases: list[dict]) -> list[dict]:
    return [
        {
            **case,
            "bench": {
                **case["bench"],
                "env": {"GOMP_SPINCOUNT": "300000", "KMP_BLOCKTIME": "200ms"},
            },
        }
        for case in cases
        if case["implementation"]["library"] != "catboost"
    ]


def generate_cases() -> list[dict]:
    return with_active_wait(gbdt_pareto.generate_cases())
