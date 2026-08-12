import importlib
import json
import logging
import os
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def get_threadpool_info():
    try:
        from threadpoolctl import threadpool_info
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to get threadpool info with "threadpoolctl" module')
        return []

    # threadpoolctl only reports libraries already loaded into *this*
    # process. A bare `import sklearn` doesn't reliably pull in its
    # OpenMP-linked Cython extensions - `get_openmp_runtime_info()` below
    # has to import this same submodule explicitly, in a subprocess, for the
    # same reason. Without this, OpenBLAS would show up (loaded as a side
    # effect of `import pandas` above) but OpenMP never would.
    try:
        from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled  # noqa: F401
    except Exception:
        pass

    threadpools = threadpool_info()
    for threadpool in threadpools:
        threadpool.pop("filepath", None)
    return threadpools


def _parse_openmp_display_env(output: str) -> dict[str, str]:
    """Parse the environment emitted by OpenMP when OMP_DISPLAY_ENV is set."""
    settings = {}
    in_display_env = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "OPENMP DISPLAY ENVIRONMENT BEGIN":
            in_display_env = True
            continue
        if stripped == "OPENMP DISPLAY ENVIRONMENT END":
            break
        if not in_display_env or "=" not in stripped:
            continue

        name, value = stripped.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        settings[name] = value
    return settings


def get_openmp_runtime_info() -> dict:
    """Collect resolved OpenMP runtime settings from a fresh Python process."""
    env = os.environ.copy()
    env["OMP_DISPLAY_ENV"] = "VERBOSE"
    # `OMP_DISPLAY_ENV` is only guaranteed to be dumped once the OpenMP
    # runtime actually initializes. GNU libgomp (the wheel-vendored build)
    # initializes eagerly on load, but the LLVM/Intel `libomp` runtime that
    # MKL builds use (via conda-forge's `_openmp_mutex=*_kmp_llvm`) only
    # initializes lazily, on the first real OpenMP call - a bare import
    # never triggers it. Calling `_openmp_effective_n_threads()`, which
    # calls `omp_get_max_threads()`, forces that initialization for every
    # build.
    import_script = (
        "from sklearn.utils._openmp_helpers import _openmp_effective_n_threads; "
        "_openmp_effective_n_threads()"
    )

    try:
        completed = subprocess.run(
            [sys.executable, "-c", import_script],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        logger.warning(f"Unable to get OpenMP runtime settings: {exc}")
        return {"error": repr(exc)}

    output = "\n".join([completed.stdout, completed.stderr])
    info = _parse_openmp_display_env(output)

    if completed.returncode != 0:
        info["returncode"] = completed.returncode
        logger.warning(
            "Unable to get complete OpenMP runtime settings: "
            f"subprocess exited with return code {completed.returncode}"
        )
    return info


def _check_output(command: list[str], cwd: str | Path | None = None) -> str | None:
    """Run a metadata command and return None when it cannot be collected."""
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            shell=False,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


@lru_cache
def _benchmark_repo_git_root() -> Path | None:
    pixi_project_root = os.environ.get("PIXI_PROJECT_ROOT")
    if pixi_project_root:
        return Path(pixi_project_root).resolve()

    git_root = _check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
    )
    if git_root is None:
        return None
    return Path(git_root).resolve()


def _git_info_for_path(path: Path) -> dict | None:
    """Return git metadata for an imported module path when it is in a checkout.

    This is best-effort environment metadata, not benchmark logic. Editable
    installs from scikit-learn source trees are the important case: Pixi can
    report that a package is installed, but the imported code's git commit is
    what makes a performance result attributable.
    """
    module_dir = path if path.is_dir() else path.parent
    git_root = _check_output(["git", "rev-parse", "--show-toplevel"], cwd=module_dir)
    if git_root is None:
        return None
    git_root = Path(git_root).resolve()
    if git_root == _benchmark_repo_git_root():
        return None

    commit = _check_output(["git", "rev-parse", "HEAD"], cwd=git_root)
    if commit is None:
        return None

    branch = _check_output(["git", "branch", "--show-current"], cwd=git_root)
    dirty = bool(
        _check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=git_root,
        )
    )

    info = {"commit": commit, "dirty": dirty}
    if branch:
        info["branch"] = branch
    describe = _check_output(
        ["git", "describe", "--tags", "--always", "--dirty"], cwd=git_root
    )
    if describe:
        info["describe"] = describe
    return info


def _distribution_name(import_name: str) -> str:
    """Map import names to their Python distribution names."""
    return {"sklearn": "scikit-learn"}.get(import_name, import_name)


def get_runtime_import_info(import_names: list[str] | None = None) -> dict:
    """Collect metadata for packages as they are imported by this environment.

    `pixi list` records the solved environment. That is not always enough for
    benchmarking source builds because an editable install can point at a local
    checkout. Runtime import metadata records the actual imported package
    version, distribution version, and git commit when the module file lives in
    a git repository.

    The default package list is intentionally short. Keep additions explicit so
    result metadata stays predictable.
    """
    if import_names is None:
        import_names = ["sklearn", "numpy", "scipy", "pandas"]

    result = {}
    for import_name in import_names:
        package_info = {}
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:
            result[import_name] = {"import_error": repr(exc)}
            continue

        runtime_version = getattr(module, "__version__", None)
        if runtime_version is not None:
            package_info["version"] = runtime_version

        dist_name = _distribution_name(import_name)
        try:
            package_info["distribution_version"] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            pass

        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            git_info = _git_info_for_path(Path(module_file).resolve())
            if git_info is not None:
                package_info["git"] = git_info

        result[import_name] = package_info
    return result


def get_software_info() -> dict:
    result = {}
    result["threadpool_info"] = get_threadpool_info()
    result["openmp_runtime_info"] = get_openmp_runtime_info()
    result["runtime_imports"] = get_runtime_import_info()

    pixi_project_root = os.environ.get("PIXI_PROJECT_ROOT")
    pixi_environment_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    result["pixi_environment_name"] = pixi_environment_name
    pixi_list = subprocess.check_output(
        [
            "pixi",
            "list",
            "--manifest-path",
            pixi_project_root,
            "--environment",
            pixi_environment_name,
            "--json",
        ],
        shell=False,
        text=True,
    )
    pixi_packages = json.loads(pixi_list)
    for pkg in pixi_packages:
        # For editable/local installs (this repo installs itself as
        # `scikit_learn_benchmarks` with source "."), pixi reports the size
        # of the live source checkout rather than a fixed artifact size. That
        # drifts with unrelated on-disk state (results/, data_cache/, ...),
        # so leaving it in would make get_software_hash() mint a new hash on
        # almost every run regardless of whether the software actually
        # changed.
        pkg.pop("size_bytes", None)
    result["pixi_packages"] = {pkg.pop("name"): pkg for pkg in pixi_packages}
    return result


def get_oneapi_devices() -> pd.DataFrame:
    try:
        import dpctl

        devices = dpctl.get_devices()
        devices = {
            device.filter_string: {
                "name": device.name,
                "vendor": device.vendor,
                "type": str(device.device_type).split(".")[1],
                "driver version": device.driver_version,
                "memory size[GB]": round(device.global_mem_size / 2**30),
            }
            for device in devices
        }
        if len(devices) > 0:
            return pd.DataFrame(devices).T
        logger.warning("dpctl device table is empty")
    except (ImportError, ModuleNotFoundError):
        logger.warning("dpctl can not be imported")
    return pd.DataFrame({"type": []})


def decode_nvml_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def get_nvml_value(getter, *args):
    try:
        return decode_nvml_value(getter(*args))
    except Exception:
        return None


def format_cuda_driver_version(version):
    if version is None:
        return None
    return f"{version // 1000}.{version % 1000 // 10}"


def get_nvidia_devices() -> pd.DataFrame:
    try:
        import pynvml
    except (ImportError, ModuleNotFoundError):
        logger.warning("pynvml can not be imported")
        return pd.DataFrame({"type": []})

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as exc:
        logger.warning(f"Unable to get NVIDIA devices with NVML: {exc}")
        return pd.DataFrame({"type": []})

    driver_version = get_nvml_value(pynvml.nvmlSystemGetDriverVersion)
    cuda_driver_version = None
    if hasattr(pynvml, "nvmlSystemGetCudaDriverVersion_v2"):
        cuda_driver_version = get_nvml_value(pynvml.nvmlSystemGetCudaDriverVersion_v2)
    elif hasattr(pynvml, "nvmlSystemGetCudaDriverVersion"):
        cuda_driver_version = get_nvml_value(pynvml.nvmlSystemGetCudaDriverVersion)
    cuda_driver_version = format_cuda_driver_version(cuda_driver_version)

    devices = {}
    for device_index in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except pynvml.NVMLError as exc:
            logger.warning(
                f"Unable to get NVIDIA device {device_index} handle with NVML: {exc}"
            )
            continue

        device_info = {
            "name": get_nvml_value(pynvml.nvmlDeviceGetName, handle),
            "vendor": "NVIDIA Corporation",
            "type": "gpu",
            "driver version": driver_version,
            "cuda driver version": cuda_driver_version,
            "index": device_index,
        }

        memory_info = get_nvml_value(pynvml.nvmlDeviceGetMemoryInfo, handle)
        if memory_info is not None:
            device_info["memory size[GB]"] = round(memory_info.total / 2**30)

        if hasattr(pynvml, "nvmlDeviceGetCudaComputeCapability"):
            compute_capability = get_nvml_value(
                pynvml.nvmlDeviceGetCudaComputeCapability, handle
            )
            if compute_capability is not None:
                device_info["cuda compute capability"] = ".".join(
                    map(str, compute_capability)
                )

        devices[f"cuda:{device_index}"] = device_info

    if len(devices) > 0:
        return pd.DataFrame(devices).T
    logger.warning("NVML device table is empty")
    return pd.DataFrame({"type": []})


def get_higher_isa(cpu_flags: str) -> str:
    ordered_sets = ["avx512", "avx2", "avx", "sse4_2", "ssse3", "sse2"]
    for isa in ordered_sets:
        if isa in cpu_flags:
            return isa
    return "unknown"


def get_hardware_info() -> dict:
    result = {}
    oneapi_devices = get_oneapi_devices()
    if len(oneapi_devices) > 0:
        logger.info(f"DPCTL listed devices:\n{oneapi_devices}\n")
    nvidia_devices = get_nvidia_devices()
    if len(nvidia_devices) > 0:
        logger.info(f"NVML listed NVIDIA devices:\n{nvidia_devices}\n")

    try:
        import joblib
        from cpuinfo import get_cpu_info

        cpu_info = get_cpu_info()
        fields_map = {
            "arch": "architecture",
            "brand_raw": "name",
            "flags": "flags",
            "count": "logical_cpus",
        }
        for key in list(cpu_info.keys()):
            value = cpu_info.pop(key)
            if key in fields_map:
                cpu_info[fields_map[key]] = value
        cpu_info["flags"] = " ".join(cpu_info["flags"])
        cpu_info["physical_cores"] = joblib.cpu_count(only_physical_cores=True)
        result["CPU"] = cpu_info
        logger.info(f"CPU name: {cpu_info['name']}")
        logger.info(f"Highest supported ISA: {get_higher_isa(cpu_info['flags']).upper()}")
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse CPU info with "cpuinfo" module')

    result["GPU(s)"] = {}
    try:
        oneapi_gpus = oneapi_devices[oneapi_devices["type"] == "gpu"]
        result["GPU(s)"].update(oneapi_gpus.T.to_dict())
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to get devices with "dpctl" module')
    try:
        nvidia_gpus = nvidia_devices[nvidia_devices["type"] == "gpu"]
        result["GPU(s)"].update(nvidia_gpus.T.to_dict())
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to get NVIDIA devices with "pynvml" module')

    try:
        import psutil

        result["RAM size[GB]"] = round(psutil.virtual_memory().total / 2**30)
        logger.info(f"RAM size[GB]: {result['RAM size[GB]']}")
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse memory info with "psutil" module')
    return result


def get_environment_info() -> dict:
    return {"hardware": get_hardware_info(), "software": get_software_info()}
