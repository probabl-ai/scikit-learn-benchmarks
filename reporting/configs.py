
import json
from pathlib import Path


HARDWARE_NAMES = {
    "0611c8": "small-laptop"
}


def read_env(kind: str, hash: str):
    assert kind in ['software', 'hardware']
    root = Path("results") / f"{kind}-envs"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing environment directory: {root}")

    candidates = []
    for path in root.glob("*.json"):
        if path.stem == hash or path.stem.endswith(f"-{hash}"):
            candidates.append(path)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {kind} environment for hash '{hash}', found {len(candidates)}"
        )
    with open(candidates[0], "r") as f:
        return json.load(f)


def _as_implementation_dict(implementation) -> dict:
    if isinstance(implementation, dict):
        return implementation
    return {
        "library": implementation.library,
        "device": implementation.device,
        "data_library": implementation.data_library,
    }


def _package_versions(env: dict, names: list[str]) -> list[dict[str, str]]:
    packages = env.get("pixi_packages", {})
    versions = []
    for name in names:
        package = packages.get(name)
        if package is not None:
            versions.append({"name": name, "version": package.get("version", "?")})
    return versions


def _threadpool_summary(env: dict) -> list[str]:
    summary = []
    for info in env.get("threadpool_info", []):
        name = info.get("prefix") or info.get("internal_api") or info.get("user_api")
        version = info.get("version")
        threads = info.get("num_threads")
        label = name
        if version:
            label += f" {version}"
        if threads is not None:
            label += f" ({threads} threads)"
        summary.append(label)
    return summary

def summarize_software_env(env: dict, implementation: dict):
    # return a small dict, ready for use in templating
    # with relevant information in the env for the given implementation:
    # include:
    # - env name
    # - implementation["library"] version, and versions of important dependencies
    # - versions of array library
    # - if relevant: summary of BLAS/threading infos
    implementation = _as_implementation_dict(implementation)
    library = implementation.get("library")
    data_library = implementation.get("data_library")
    package_names = [
        "python",
        "scikit-learn",
        "scikit_learn_intelex",
        "scikit_learn_intelex_gpu",
        "daal",
        "numpy",
        "scipy",
        "pandas",
        "array_api_compat",
        "dpnp",
        "torch",
        "cupy",
    ]
    if library and library not in package_names:
        package_names.insert(1, library)
    if data_library and data_library not in package_names:
        package_names.append(data_library)

    device = implementation.get("device") or "default"
    return {
        "environment": env.get("pixi_environment_name", "?"),
        "implementation": implementation,
        "implementation_label": (
            library if device == "default" else f"{library} on {device}"
        ),
        "packages": _package_versions(env, package_names),
        "threadpools": _threadpool_summary(env),
    }

def summarize_hardware_env(env: dict):
    # same for hardware, but independant of implem
    cpu = env.get("CPU", {})
    gpus = env.get("GPU(s)", {})
    return {
        "cpu_name": cpu.get("name", "?"),
        "architecture": cpu.get("architecture", "?"),
        "logical_cpus": cpu.get("logical_cpus", "?"),
        "ram_gb": env.get("RAM size[GB]", "?"),
        "gpus": [
            {
                "id": device_id,
                "name": gpu.get("name", "?"),
                "vendor": gpu.get("vendor", "?"),
                "driver": gpu.get("driver version", "?"),
                "memory_gb": gpu.get("memory size[GB]", "?"),
            }
            for device_id, gpu in gpus.items()
        ],
    }
