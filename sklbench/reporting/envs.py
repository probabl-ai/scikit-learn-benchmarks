
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
# "sklearn" was the pixi environment's name before the 2026-08-13 pixi.toml
# rename to "sklearn-pypi"; results captured before that rename still record
# the old name, so both must match.
VANILLA_SKLEARN_PIXI_ENVS = {"sklearn", "sklearn-pypi"}


@lru_cache(maxsize=None)
def is_vanilla_sklearn(software_hash: str) -> bool:
    return read_env("software", software_hash)["pixi_environment_name"] in VANILLA_SKLEARN_PIXI_ENVS


# One-off sklearn-conda run with GOMP_SPINCOUNT manually raised to
# 10_000_000 (vs. libgomp's 300_000 default), to probe busy-wait tuning.
# The pixi env name alone can't distinguish it from the regular
# sklearn-conda run, and no more of these are planned, so it's special-cased
# by hash here instead of adding a generic env-variant concept.
_SPINCOUNT_VARIANT_SOFTWARE_HASH = "a2680a"


@lru_cache(maxsize=None)
def software_build_name(software_hash: str) -> str:
    name = read_env("software", software_hash)["pixi_environment_name"]
    if software_hash == _SPINCOUNT_VARIANT_SOFTWARE_HASH:
        return f"{name}-10M-spincount"
    return name


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


@lru_cache(maxsize=None)
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
        # pixi keys pypi packages by their normalized (underscore) name but
        # conda packages by their distribution (hyphenated) name, so the same
        # logical package (e.g. scikit-learn) can appear under either spelling
        # depending on whether it was installed via pip or conda.
        package = packages.get(name) or packages.get(name.replace("-", "_"))
        if package is None:
            continue

        kind = package.get("kind")
        summary = {
            "name": name,
            "version": package.get("version", "?"),
            "kind": "conda-forge" if kind == "conda" else kind,
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


# threadpoolctl (`_threadpool_summary` above) only reports BLAS/LAPACK
# threadpools it recognizes by loaded library, not the OpenMP runtime
# sklearn's own Cython loops use - that's captured separately, by dumping
# `OMP_DISPLAY_ENV=VERBOSE`'s output (env-var name -> value, some prefixed
# with e.g. "[host] " depending on the libgomp/libomp version). Presence of
# GOMP_*/KMP_* vars is what actually tells GNU libgomp apart from
# Intel/LLVM's OpenMP - that's the "which OpenMP build" question this exists
# to answer.
_OPENMP_SPEC_VERSIONS = {
    "201107": "3.1",
    "201307": "4.0",
    "201511": "4.5",
    "201811": "5.0",
    "202011": "5.1",
    "202111": "5.2",
}


def _openmp_runtime_family(info: dict) -> str:
    var_names = {key.rsplit(" ", 1)[-1] for key in info}
    if any(name.startswith("GOMP_") for name in var_names):
        return "GNU libgomp"
    if any(name.startswith("KMP_") for name in var_names):
        return "Intel/LLVM OpenMP"
    return "unknown OpenMP runtime"


def _openmp_env_value(info: dict, var_name: str) -> str | None:
    for key, value in info.items():
        if key == var_name or key.endswith(f" {var_name}"):
            # Intel/LLVM's OMP_DISPLAY_ENV dump sometimes leaves a stray,
            # unbalanced quote around values (e.g. "'0ms") that the capture
            # side's quote-stripping (env.py's `_parse_openmp_display_env`,
            # which only strips a quote present on *both* ends) doesn't
            # catch.
            return value.strip("'\"") if isinstance(value, str) else value
    return None


def _openmp_summary(env: dict) -> list[str]:
    info = env.get("openmp_runtime_info")
    if not info:
        return []
    label = _openmp_runtime_family(info)
    spec_code = _openmp_env_value(info, "_OPENMP")
    if spec_code:
        spec_version = _OPENMP_SPEC_VERSIONS.get(spec_code)
        label += f" (OpenMP {spec_version})" if spec_version else f" (spec {spec_code})"
    summary = [label]
    # Busy-wait tuning knob, one or the other depending on runtime family -
    # how long idle threads spin before sleeping, a common source of
    # thread-scaling differences between builds.
    gomp_spincount = _openmp_env_value(info, "GOMP_SPINCOUNT")
    if gomp_spincount:
        summary.append(f"GOMP_SPINCOUNT: {gomp_spincount}")
    kmp_blocktime = _openmp_env_value(info, "KMP_BLOCKTIME")
    if kmp_blocktime:
        summary.append(f"KMP_BLOCKTIME: {kmp_blocktime}")
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
        "openmp": _openmp_summary(env),
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
