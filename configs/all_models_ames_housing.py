"""Temp config: all_models.py cases filtered to the ames_housing dataset only."""
from all_models import generate_cases as generate_all_cases


def generate_cases() -> list[dict]:
    cases = generate_all_cases()
    return [case for case in cases if case.data.dataset == "ames_housing"]
