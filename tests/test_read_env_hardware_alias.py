import json

from sklbench.reporting import envs


def _write_hardware_env(root, hash, cpu):
    env_dir = root / "results" / "hardware-envs"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / f"{hash}.json").write_text(json.dumps({"cpu": cpu}))


def test_read_env_falls_back_to_an_alias_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(envs, "HARDWARE_HASH_ALIASES", {"newhsh": "oldhsh"})
    _write_hardware_env(tmp_path, "newhsh", "new")

    assert envs.read_env("hardware", "oldhsh") == {"cpu": "new"}


def test_read_env_prefers_the_canonical_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(envs, "HARDWARE_HASH_ALIASES", {"newhsh": "oldhsh"})
    _write_hardware_env(tmp_path, "newhsh", "new")
    _write_hardware_env(tmp_path, "oldhsh", "old")

    assert envs.read_env("hardware", "oldhsh") == {"cpu": "old"}
