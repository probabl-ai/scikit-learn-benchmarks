"""Tests for configs/_numa.py. Imported directly (not a package under
sklbench/) the same way configs/synthetic_linear.py imports it - by putting
configs/ on sys.path.
"""
import importlib
import sys
from pathlib import Path

import pytest


CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


@pytest.fixture
def numa(monkeypatch):
    inserted = CONFIGS_DIR not in sys.path
    if inserted:
        sys.path.insert(0, CONFIGS_DIR)
    try:
        module = importlib.import_module("_numa")
        importlib.reload(module)
        yield module
    finally:
        if inserted:
            sys.path.remove(CONFIGS_DIR)


def _make_fake_numa_sysfs(tmp_path, node_cpulists: dict[int, str]):
    root = tmp_path / "node"
    root.mkdir()
    for node, cpulist in node_cpulists.items():
        node_dir = root / f"node{node}"
        node_dir.mkdir()
        (node_dir / "cpulist").write_text(cpulist + "\n")
    return root


def test_numa_node_count_reads_node_directories(numa, tmp_path, monkeypatch):
    fake_root = _make_fake_numa_sysfs(tmp_path, {0: "0-21", 1: "22-43"})
    monkeypatch.setattr(numa, "NUMA_NODES_SYSFS", fake_root)

    assert numa.numa_node_count() == 2


def test_numa_node_count_defaults_to_one_without_sysfs(numa, tmp_path, monkeypatch):
    monkeypatch.setattr(numa, "NUMA_NODES_SYSFS", tmp_path / "does-not-exist")

    assert numa.numa_node_count() == 1


def test_taskset_for_numa_node_reads_cpulist(numa, tmp_path, monkeypatch):
    fake_root = _make_fake_numa_sysfs(tmp_path, {3: "0-21,172-193"})
    monkeypatch.setattr(numa, "NUMA_NODES_SYSFS", fake_root)

    assert numa.taskset_for_numa_node(3) == "0-21,172-193"


def test_auto_numa_taskset_is_none_on_a_single_node_host(numa, tmp_path, monkeypatch):
    fake_root = _make_fake_numa_sysfs(tmp_path, {0: "0-343"})
    monkeypatch.setattr(numa, "NUMA_NODES_SYSFS", fake_root)

    assert numa.auto_numa_taskset() is None


def test_auto_numa_taskset_pins_to_the_requested_node_on_a_multi_node_host(
    numa, tmp_path, monkeypatch
):
    fake_root = _make_fake_numa_sysfs(tmp_path, {0: "0-21", 1: "22-43"})
    monkeypatch.setattr(numa, "NUMA_NODES_SYSFS", fake_root)

    assert numa.auto_numa_taskset(node=1) == "22-43"
