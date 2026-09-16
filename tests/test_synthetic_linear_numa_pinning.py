import importlib.util
import sys
from pathlib import Path

import pytest

from sklbench.config import validate_case


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(scope="module")
def synthetic_linear():
    """Import configs/synthetic_linear.py the same way
    `sklbench.config.loader._load_module_from_path` does - it's a script
    meant to be run with `configs/` on `sys.path` (for its own
    `from _common import ...` / `from _numa import ...`), not an importable
    package.
    """
    path = CONFIGS_DIR / "synthetic_linear.py"
    spec = importlib.util.spec_from_file_location("synthetic_linear", path)
    module = importlib.util.module_from_spec(spec)
    configs_dir = str(CONFIGS_DIR)
    inserted = configs_dir not in sys.path
    if inserted:
        sys.path.insert(0, configs_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(configs_dir)
    return module


def _a_case(synthetic_linear):
    return synthetic_linear.generate_cases(tier="test")[0]


def test_with_numa_pinning_sets_cpu_affinity_from_the_node(synthetic_linear, monkeypatch):
    monkeypatch.setattr(synthetic_linear, "auto_numa_cpu_affinity", lambda node=0: [0, 1, 2])

    case = synthetic_linear.with_numa_pinning(_a_case(synthetic_linear))

    assert case["bench"]["cpu_affinity"] == [0, 1, 2]


def test_with_numa_pinning_is_a_noop_on_a_single_node_host(synthetic_linear, monkeypatch):
    monkeypatch.setattr(synthetic_linear, "auto_numa_cpu_affinity", lambda node=0: None)

    case = _a_case(synthetic_linear)
    pinned = synthetic_linear.with_numa_pinning(case)

    assert pinned == case
    assert "cpu_affinity" not in pinned.get("bench", {})


def test_with_numa_pinning_preserves_other_bench_fields(synthetic_linear, monkeypatch):
    monkeypatch.setattr(synthetic_linear, "auto_numa_cpu_affinity", lambda node=0: [0, 1, 2])

    case = _a_case(synthetic_linear)
    case["bench"] = {**case.get("bench", {}), "n_runs": 3}

    pinned = synthetic_linear.with_numa_pinning(case)

    assert pinned["bench"]["n_runs"] == 3
    assert pinned["bench"]["cpu_affinity"] == [0, 1, 2]


def test_with_numa_pinning_refuses_to_override_an_existing_cpu_affinity(synthetic_linear, monkeypatch):
    monkeypatch.setattr(synthetic_linear, "auto_numa_cpu_affinity", lambda node=0: [0, 1, 2])

    case = _a_case(synthetic_linear)
    case["bench"] = {**case.get("bench", {}), "cpu_affinity": [0, 3]}

    with pytest.raises(ValueError, match="already sets bench.cpu_affinity"):
        synthetic_linear.with_numa_pinning(case)


def test_with_numa_pinning_produces_a_valid_case(synthetic_linear, monkeypatch):
    monkeypatch.setattr(synthetic_linear, "auto_numa_cpu_affinity", lambda node=0: [0, 1, 2])

    pinned = synthetic_linear.with_numa_pinning(_a_case(synthetic_linear))

    validated = validate_case(pinned)

    assert validated.bench.cpu_affinity == [0, 1, 2]
