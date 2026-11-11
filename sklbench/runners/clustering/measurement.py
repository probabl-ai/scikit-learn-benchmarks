"""Device placement and memory/GPU sampling around the timed k-means fits."""

from __future__ import annotations

import threading


class Device:
    """Where the fits read the data: host memory (``data_library`` None), or a CUDA
    device through CuPy or PyTorch."""

    def __init__(self, data_library: str | None):
        self.data_library = data_library
        if data_library == "cupy":
            import cupy

            self.cupy = cupy
        elif data_library == "torch":
            import torch

            self.torch = torch
        elif data_library is not None:
            raise ValueError(f"unsupported data_library {data_library!r}")

    @property
    def is_gpu(self) -> bool:
        return self.data_library is not None

    def put(self, X):
        if self.data_library == "cupy":
            return self.cupy.asarray(X)
        if self.data_library == "torch":
            return self.torch.from_numpy(X).to("cuda")
        return X

    def sync(self):
        if self.data_library == "cupy":
            self.cupy.cuda.Device().synchronize()
        elif self.data_library == "torch":
            self.torch.cuda.synchronize()

    def release(self):
        """Empty the memory pools, so that the peak memory of a fit starts from
        what the data needs."""
        if self.data_library == "cupy":
            self.cupy.get_default_memory_pool().free_all_blocks()
        elif self.data_library == "torch":
            self.torch.cuda.empty_cache()


def reset_peak_rss() -> bool:
    """Reset the kernel's peak RSS counter (VmHWM) of this process (Linux only)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _status_bytes(field: str) -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def rss_bytes() -> int | None:
    return _status_bytes("VmRSS:")


def peak_rss_bytes() -> int | None:
    return _status_bytes("VmHWM:")


class GPUSampler:
    """Samples device memory, SM clock, power and temperature of GPU 0 in a thread.

    The SM clock shows the power cap at work: an L4 runs long fits well below its
    maximum clock. Peak memory is the device-wide used memory minus the value at
    ``start`` (the machine is locked), sampled every ``interval`` seconds, so
    shorter allocations can be missed.
    """

    def __init__(self, interval=0.02):
        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _memory_used(self) -> int:
        return self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used

    def _sample(self) -> dict:
        nvml, handle = self.nvml, self.handle
        return {
            "memory_used": self._memory_used(),
            "sm_clock_mhz": nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
            "power_w": nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "temperature_c": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
        }

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval)

    def start(self):
        self.baseline = self._memory_used()
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        self.samples.append(self._sample())
        samples = self.samples
        out = {"peak_memory_bytes": max(s["memory_used"] for s in samples) - self.baseline}
        for key in ("sm_clock_mhz", "power_w"):
            out[f"gpu_{key}_mean"] = round(sum(s[key] for s in samples) / len(samples), 1)
        out["gpu_temperature_c_max"] = max(s["temperature_c"] for s in samples)
        return out
