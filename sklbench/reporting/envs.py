
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

from .matching import Implementation


def read_env(kind: str, hash: str):
    assert kind in ['software', 'hardware']
    path = Path("results") / f"{kind}-envs" / f"{hash}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Expected {kind} environment file: {path}")
    with open(path, "r") as f:
        return json.load(f)


# Several pixi envs build sklearn differently (pip, conda, MKL, OpenBLAS...);
# results from all of them report implementation.short_name == "sklearn".
# Dashboards that compare implementations against a single sklearn baseline
# should restrict to this one to avoid ambiguous matches.
VANILLA_SKLEARN_PIXI_ENV = "sklearn"


@lru_cache(maxsize=None)
def is_vanilla_sklearn(software_hash: str) -> bool:
    return read_env("software", software_hash)["pixi_environment_name"] == VANILLA_SKLEARN_PIXI_ENV


DEFAULT_SOURCE = {
    "conda": "https://conda.anaconda.org/conda-forge",
    "pypi": "https://pypi.org/simple",
}

RESULTS_REPOSITORY = "probabl-ai/scikit-learn-benchmarks"
DEFAULT_RESULTS_REF = "refs/heads/main"
JSON_VIEWER_BASE_URL = (
    "https://flamegraph-viewer-839234844562.europe-west1.run.app/json"
)
FLAMEGRAPH_VIEWER_BASE_URL = (
    "https://flamegraph-viewer-839234844562.europe-west1.run.app/"
)


def _current_results_ref() -> str:
    explicit_ref = os.environ.get("SKLBENCH_RESULTS_REF")
    if explicit_ref:
        return explicit_ref.strip("/")

    github_ref = os.environ.get("GITHUB_REF")
    if github_ref:
        return github_ref.strip("/")

    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        branch = ""
    if branch:
        return f"refs/heads/{branch}"

    return DEFAULT_RESULTS_REF


def github_raw_url(path: Path | str) -> str:
    path = Path(path).as_posix().lstrip("/")
    return (
        f"https://github.com/{RESULTS_REPOSITORY}/raw/{_current_results_ref()}/"
        f"{path}"
    )


def json_viewer_url(path: Path | str) -> str:
    return f"{JSON_VIEWER_BASE_URL}?url={quote(github_raw_url(path), safe='')}"


def profile_viewer_url(path: Path | str) -> str:
    return f"{FLAMEGRAPH_VIEWER_BASE_URL}?url={quote(github_raw_url(path), safe='')}"


def software_env_json_url(software_hash: str) -> str:
    raw_url = github_raw_url(f"results/software-envs/{software_hash}.json")
    return f"{JSON_VIEWER_BASE_URL}?url={quote(raw_url, safe='')}"


def _package_versions(env: dict, names: list[str]) -> list[dict[str, str]]:
    packages = env.get("pixi_packages", {})
    versions = []
    for name in names:
        package = packages.get(name)
        if package is None:
            continue

        kind = package.get("kind")
        summary = {
            "name": name,
            "version": package.get("version", "?"),
            "kind": "pip" if kind == "pypi" else kind,
        }

        default_source = DEFAULT_SOURCE.get(kind)
        if default_source and default_source != package.get("source"):
            summary["source"] = package.get("source")

        versions.append(summary)

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


def _implementation_package_names(implementation: Implementation) -> list[str]:
    library = implementation.library
    data_library = implementation.data_library

    if library == "sklearn" and data_library:
        package_names = ["scikit-learn", data_library]
        return package_names

    if library == "sklearn":
        return ["scikit-learn", "numpy", "scipy", "pandas"]

    if library == "sklearnex":
        package_names = ["scikit-learn", "scikit_learn_intelex"]
        if implementation.device == "gpu":
            package_names += ["scikit_learn_intelex_gpu", "dpnp"]
        package_names.append("daal")
        return package_names

    return [name for name in [library, data_library] if name]


def summarize_software_env(
    env: dict,
    implementation: Implementation,
    *,
    software_hash: str | None = None,
):
    # return a small dict, ready for use in templating
    # with relevant information in the env for the given implementation:
    # include:
    # - env name
    # - implementation["library"] version, and versions of important dependencies
    # - versions of array library
    # - if relevant: summary of BLAS/threading infos
    package_names = _implementation_package_names(implementation)

    python_version, = _package_versions(env, ["python"])

    out = {
        "name": implementation.short_name,
        "python_version": python_version["version"],
        "packages": _package_versions(env, package_names),
        "threadpools": _threadpool_summary(env),
    }
    if software_hash is not None:
        out["software_env_json_url"] = software_env_json_url(software_hash)
    if implementation.library == "sklearn" and implementation.data_library:
        out["array_api_docs_url"] = "https://scikit-learn.org/stable/modules/array_api.html"
    return out


def summarize_hardware_env(env: dict):
    # same for hardware, but independant of implem
    cpu = env.get("CPU", {})
    gpus = env.get("GPU(s)", {})

    drivers = {k.split(":")[0] for k in gpus}

    if drivers == {"level_zero", "opencl"}:
        gpus = {k: v for k, v in gpus.items() if k.startswith("level_zero")}

    return {
        "cpu_name": cpu.get("name", "?"),
        "architecture": cpu.get("architecture", "?"),
        "logical_cpus": cpu.get("logical_cpus", "?"),
        "physical_cores": cpu.get("physical_cores", "?"),
        "ram_gb": env.get("RAM size[GB]", "?"),
        "gpus": [
            {
                "id": device_id,
                "name": gpu.get("name", "?"),
                "memory_gb": gpu.get("memory size[GB]", "?"),
            }
            for device_id, gpu in gpus.items()
        ],
    }
