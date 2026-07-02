import gc
import logging
import subprocess
import sys
import threading
import timeit
from math import ceil, sqrt
from time import sleep

import joblib
import numpy as np
import psutil
from cpuinfo import get_cpu_info

from ..config import Bench

logger = logging.getLogger(__name__)

try:
    import pynvml

    try:
        pynvml.nvmlInit()
        nvml_is_available = True
    except pynvml.NVMLError:
        nvml_is_available = False
except (ImportError, ModuleNotFoundError):
    nvml_is_available = False


def _get_n_from_cache_size():
    cache_size = 0
    cpu_info = get_cpu_info()
    if "l3_cache_size" in cpu_info:
        cache_size += cpu_info["l3_cache_size"]
    if "l2_cache_size" in cpu_info:
        cache_size += cpu_info["l2_cache_size"] * psutil.cpu_count(logical=False)
    n_sockets = _get_number_of_sockets()
    return ceil(sqrt(n_sockets * cache_size / 8))


def read_output_from_command(
    command: str, timeout: float | None = None
) -> tuple[int, str, str]:
    try:
        res = subprocess.run(
            command.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timeout_message = f"Command timed out after {timeout} seconds."
        stderr = f"{stderr.strip()}\n{timeout_message}".strip()
        return -9, stdout.strip(), stderr
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def _get_number_of_sockets():
    if sys.platform == "win32":
        try:
            result = subprocess.check_output(
                ["wmic", "cpu", "get", "DeviceID"], shell=False, text=True
            )
            return len([line for line in result.split("\n") if line.startswith("CPU")])
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            logger.warning("Unable to get number of sockets via wmic")
            return 1
    if sys.platform == "linux":
        try:
            _, lscpu_text, _ = read_output_from_command("lscpu")
            for line in lscpu_text.split("\n"):
                if "Socket(s):" in line:
                    return int(line.split(":")[1].strip())
            logger.warning("Unable to find Socket(s) information in lscpu output")
        except (FileNotFoundError, ValueError, IndexError):
            logger.warning("Unable to get number of sockets via lscpu")
        return 1
    logger.warning("Unable to get number of sockets due to unknown sys.platform")
    return 1


def _flush_cache(n: int | None = None):
    if n is None:
        n = _get_n_from_cache_size()
    np.matmul(np.random.rand(n, n), np.random.rand(n, n))


def _get_ram_usage():
    return psutil.Process().memory_info().rss


def _get_vram_usage():
    pid = psutil.Process().pid

    device_count = pynvml.nvmlDeviceGetCount()
    vram_usage = 0
    for device_index in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for process in process_info:
            if process.pid == pid:
                vram_usage += process.usedGpuMemory
    return vram_usage


def _monitor_memory_usage(
    interval: float,
    memory_profiles: dict[str, list],
    stop_event,
    enable_nvml_profiling: bool,
):
    while not stop_event.is_set():
        memory_profiles["RAM"].append(_get_ram_usage())
        if enable_nvml_profiling:
            memory_profiles["VRAM"].append(_get_vram_usage())
        sleep(interval)


def measure_perf(
    func,
    *args,
    bench_params: Bench = None,
    **kwargs,
):
    """
    Measure performance metrics of a function execution.

    Executes the given function and collects performance metrics including
    execution time and optionally CPU load, memory usage (RAM and VRAM),
    with support for cache flushing and profiling integration.

    Parameters
    ----------
    func : callable
        The function to measure.
    *args
        Positional arguments to pass to func.
    bench_params : Bench, optional
        Benchmark configuration parameters controlling which metrics to collect
        and runtime options such as cache flushing and garbage collection.
        If None, default Bench() parameters are used.
    **kwargs
        Keyword arguments to pass to func.

    Returns
    -------
    tuple
        A tuple of (time_ms, perf_metrics) where:
        - time_ms : float
            Execution time in milliseconds.
        - perf_metrics : dict
            Dictionary containing optional performance metrics:
            - "peak RAM usage[MB]" : list of floats (if memory_profile enabled)
            - "peak VRAM usage[MB]" : list of floats (if NVML profiling enabled)
            - "cpu load[%]" : list of floats (if cpu_profile enabled)
    """
    if bench_params is None:
        bench_params = Bench()  # use defaults
    enable_cache_flushing = bench_params.flush_cache
    enable_garbage_collection = bench_params.gc_collect
    enable_cpu_profiling = bench_params.cpu_profile
    enable_memory_profiling = bench_params.memory_profile
    memory_profiling_interval = bench_params.memory_profiling_interval
    enable_nvml_profiling = False

    if enable_cpu_profiling:
        cpu_loads = []
    if enable_memory_profiling:
        memory_peaks = {"RAM": []}
        if enable_nvml_profiling:
            memory_peaks["VRAM"] = []

    if enable_cache_flushing:
        _flush_cache()
    if enable_memory_profiling:
        memory_profiles = {"RAM": []}
        if enable_nvml_profiling:
            memory_profiles["VRAM"] = []
        profiling_stop_event = threading.Event()
        profiling_thread = threading.Thread(
            target=_monitor_memory_usage,
            args=(
                memory_profiling_interval,
                memory_profiles,
                profiling_stop_event,
                enable_nvml_profiling,
            ),
        )
        profiling_thread.start()
    if enable_cpu_profiling:
        psutil.cpu_percent(interval=None)

    t0 = timeit.default_timer()
    _ = func(*args, **kwargs)
    t1 = timeit.default_timer()

    if enable_cpu_profiling:
        cpu_loads.append(psutil.cpu_percent(interval=None))
    if enable_memory_profiling:
        profiling_stop_event.set()
        profiling_thread.join()
        memory_peaks["RAM"].append(max(memory_profiles["RAM"]))
        if enable_nvml_profiling:
            memory_peaks["VRAM"].append(max(memory_profiles["VRAM"]))

    time_ms = 1000 * (t1 - t0)
    if enable_garbage_collection:
        gc.collect()

    perf_metrics = {
        "n_detected_physical_cores": joblib.cpu_count(only_physical_cores=True),
        "n_detected_logical_cpus": joblib.cpu_count(only_physical_cores=False),
    }
    if enable_memory_profiling:
        perf_metrics["peak RAM usage[MB]"] = [
            memory_peak / 2**20 for memory_peak in memory_peaks["RAM"]
        ]
        if enable_nvml_profiling:
            perf_metrics["peak VRAM usage[MB]"] = [
                memory_peak / 2**20 for memory_peak in memory_peaks["VRAM"]
            ]
    if enable_cpu_profiling:
        perf_metrics["cpu load[%]"] = cpu_loads

    return time_ms, perf_metrics
