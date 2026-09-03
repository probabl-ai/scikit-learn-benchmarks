import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


def _sample_temperatures_c() -> dict[str, float]:
    """Flatten `psutil.sensors_temperatures()` into `{"chip:label": celsius}`.

    Not available on every platform (e.g. macOS, some containers/VMs), so
    every step here is best-effort and returns `{}` rather than raising.
    """
    get_temperatures = getattr(psutil, "sensors_temperatures", None)
    if get_temperatures is None:
        return {}
    try:
        chips = get_temperatures()
    except Exception:
        return {}
    temperatures = {}
    for chip_name, readings in chips.items():
        for index, reading in enumerate(readings):
            label = reading.label or str(index)
            temperatures[f"{chip_name}:{label}"] = reading.current
    return temperatures


def _sample_cpu_freq_mhz() -> list[float] | None:
    try:
        per_cpu = psutil.cpu_freq(percpu=True)
    except Exception:
        return None
    if not per_cpu:
        return None
    return [freq.current for freq in per_cpu]


def _sample_load_avg() -> list[float] | None:
    try:
        return list(psutil.getloadavg())
    except (AttributeError, OSError):
        return None


def _sample_memory() -> dict | None:
    """System-wide (not per-process) RAM/swap usage.

    Deliberately system-wide rather than the orchestrator process's own RSS:
    the actual benchmarked work runs in a separate `taskset`d subprocess (see
    `commands.py`), so the orchestrator's own memory use wouldn't reflect it
    anyway. A 2s cadence can't catch a brief peak the way a per-call
    high-frequency profiler could, but a large allocation or a
    swap-thrashing machine stays visibly elevated for multiple samples in a
    row - active swapping in particular is a plausible cause of the kind of
    >100x wall time blowups this monitor exists to help diagnose, and this
    is the cheapest way to see whether it was happening.
    """
    try:
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
    except Exception:
        return None
    return {
        "used_percent": virtual_memory.percent,
        "available_mb": virtual_memory.available / 2**20,
        "swap_used_percent": swap_memory.percent,
        "swap_used_mb": swap_memory.used / 2**20,
    }


class SystemMonitor:
    """Background sampler (CPU load/frequency/temperature, system-wide
    RAM/swap) that runs for the whole orchestrator session rather than
    being scoped to one case's fit/predict call.

    A pathological repeat (e.g. a laptop hybrid P/E-core scheduling stall)
    can be buried inside a single subprocess call that the per-case JSON
    record has no visibility into beyond a wall-clock `time_ms`. Sampling
    independently of case boundaries, on a fixed wall-clock cadence, lets a
    later investigation line the telemetry timestamps up against a
    suspicious repeat's timestamp instead of having to reproduce it live.
    """

    def __init__(self, output_path: Path, interval: float = 2.0):
        self._output_path = output_path
        self._interval = interval
        self.case_index: int | None = None
        self.case_name: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # In-memory buffer backing `samples_between()`, separate from the
        # on-disk JSONL (which stays a complete, uninterrupted log of the
        # whole session - warmup, gaps between cases, everything). Entries
        # are `(sampled_at, sample)`; `samples_between` both reads and
        # prunes it (see its docstring for why that's safe).
        self._samples: list[tuple[datetime, dict]] = []
        self._samples_lock = threading.Lock()

    def set_current_case(self, index: int, name: str) -> None:
        # Plain attribute assignment, read back from the sampling thread -
        # no lock needed, the GIL makes each of these assignments/reads
        # atomic and a torn read (old index with new name, say) is harmless
        # for a debugging aid like this.
        self.case_index = index
        self.case_name = name

    def samples_between(self, start: datetime, end: datetime) -> list[dict]:
        """Buffered samples with `start <= sampled_at <= end`, for embedding
        into one case's saved record.

        Also discards every buffered sample with `sampled_at <= end`. The
        orchestrator calls this once per case, right after that case
        finishes, and cases run strictly sequentially - so nothing at or
        before this case's end time can ever be relevant to a later query.
        Without this the buffer would grow unbounded over a multi-hour
        session; with it, memory use stays proportional to one case's
        duration instead.
        """
        with self._samples_lock:
            matched = [sample for sampled_at, sample in self._samples if start <= sampled_at <= end]
            self._samples = [
                (sampled_at, sample) for sampled_at, sample in self._samples if sampled_at > end
            ]
        return matched

    def _sample(self, sampled_at: datetime) -> dict:
        sample = {
            "timestamp": sampled_at.isoformat(timespec="milliseconds"),
            "case_index": self.case_index,
            "case_name": self.case_name,
        }
        try:
            sample["cpu_percent"] = psutil.cpu_percent(percpu=True, interval=None)
        except Exception:
            pass
        cpu_freq_mhz = _sample_cpu_freq_mhz()
        if cpu_freq_mhz is not None:
            sample["cpu_freq_mhz"] = cpu_freq_mhz
        temperatures_c = _sample_temperatures_c()
        if temperatures_c:
            sample["temperatures_c"] = temperatures_c
        load_avg = _sample_load_avg()
        if load_avg is not None:
            sample["load_avg"] = load_avg
        memory = _sample_memory()
        if memory is not None:
            sample["memory"] = memory
        return sample

    def _run(self) -> None:
        # Primes psutil's internal per-call delta baseline so the first
        # real sample reflects usage since monitoring started rather than
        # since interpreter startup.
        psutil.cpu_percent(percpu=True, interval=None)
        with self._output_path.open("a", encoding="utf-8") as fp:
            while not self._stop_event.wait(self._interval):
                try:
                    sampled_at = datetime.now(timezone.utc)
                    sample = self._sample(sampled_at)
                    fp.write(json.dumps(sample) + "\n")
                    fp.flush()
                    with self._samples_lock:
                        self._samples.append((sampled_at, sample))
                except Exception:
                    logger.warning("System telemetry sample failed", exc_info=True)

    def start(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 5)
