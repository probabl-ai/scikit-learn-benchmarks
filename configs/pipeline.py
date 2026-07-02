import joblib
import numpy as np

from sklbench.config import PipelineCase, PipelineData, PipelineRun
from ._scaling import get_n_cores_list


def generate_cases() -> list[PipelineCase]:
    return [
        PipelineCase(
            bench={"n_runs": 1, "time_limit": 600},
            data=PipelineData(openml_data_id=42165),
            run=PipelineRun(n_jobs=n_jobs),
        )
        for n_jobs in get_n_cores_list()
    ]
