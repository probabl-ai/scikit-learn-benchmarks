import joblib
import numpy as np

from sklbench.config import PipelineCase, PipelineData, PipelineRun


def _n_jobs_values() -> list[int]:
    cpu_count = joblib.cpu_count(only_physical_cores=True)
    return [int(2**power) for power in range(int(np.log2(cpu_count)) + 1)]


def generate_cases() -> list[PipelineCase]:
    return [
        PipelineCase(
            bench={"n_runs": 1, "time_limit": 600},
            data=PipelineData(openml_data_id=42165),
            run=PipelineRun(n_jobs=n_jobs),
        )
        for n_jobs in _n_jobs_values()
    ]
