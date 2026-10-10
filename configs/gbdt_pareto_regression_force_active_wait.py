"""`gbdt_pareto_regression.py` with OpenMP active wait forced, see
`gbdt_pareto_force_active_wait.py`."""
import configs.gbdt_pareto_force_active_wait as gbdt_pareto_force_active_wait
import configs.gbdt_pareto_regression as gbdt_pareto_regression


def generate_cases() -> list[dict]:
    return gbdt_pareto_force_active_wait.with_active_wait(
        gbdt_pareto_regression.generate_cases()
    )
