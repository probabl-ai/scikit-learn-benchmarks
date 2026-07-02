from sklbench.config import PipelineCase, PipelineData, PipelineRun


def generate_cases() -> list[PipelineCase]:
    return [
        PipelineCase(
            bench={"n_runs": 1, "time_limit": 600},
            data=PipelineData(openml_data_id=42165),
            run=PipelineRun(),
        )
    ]
