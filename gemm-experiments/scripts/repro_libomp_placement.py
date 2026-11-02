"""Reproducer: MKL threads share physical cores under conda-forge's LLVM libomp.

Runs X @ X (float64, 4096x4096) in fresh processes, one MKL thread per
physical core of this process's affinity mask, and samples every 20 ms on
which CPU each running thread of the process sits. Prints GFLOP/s and the
average number of distinct CPUs and physical cores used by the running
threads, for the default settings and a few alternatives.

    pixi exec -s python=3.12 -s numpy -s "libblas=*=*mkl" -- python repro_libomp_placement.py
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def core_of(cpu):
    topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return (topo / "physical_package_id").read_text().strip(), (topo / "core_id").read_text().strip()


def child():
    import numpy as np

    n = 4096
    a = np.random.default_rng(0).standard_normal((n, n))
    a @ a  # warm up the thread pool
    samples, stop = [], threading.Event()

    def sample():
        pid, me = os.getpid(), str(threading.get_native_id())
        while not stop.wait(0.02):
            cpus = []
            for tid in os.listdir(f"/proc/{pid}/task"):
                if tid == me:
                    continue
                try:
                    stat = Path(f"/proc/{pid}/task/{tid}/stat").read_text()
                except OSError:
                    continue
                fields = stat.rsplit(")", 1)[1].split()
                if fields[0] == "R":  # state; field 39 (processor) is fields[36]
                    cpus.append(int(fields[36]))
            samples.append((len(cpus), len(set(cpus)), len({core_of(c) for c in cpus})))

    t = threading.Thread(target=sample)
    t.start()
    times = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3:
        t1 = time.perf_counter()
        a @ a
        times.append(time.perf_counter() - t1)
    stop.set()
    t.join()
    times.sort()
    gflops = 2 * n**3 / times[len(times) // 2] / 1e9
    k = len(samples)
    run, cpus, cores = (sum(s[i] for s in samples) / k for i in range(3))
    print(f"{gflops:8.0f} GFLOP/s   running {run:5.1f}   cpus {cpus:5.1f}   cores {cores:5.1f}",
          flush=True)


def main():
    cpus = sorted(os.sched_getaffinity(0))
    ncores = len({core_of(c) for c in cpus})
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else ncores
    meta = Path(sys.prefix) / "conda-meta"
    pkgs = sorted(f.stem for f in meta.glob("*.json")
                  if f.stem.split("-")[0] in ("numpy", "mkl", "llvm", "libblas", "_openmp"))
    print("packages:", ", ".join(pkgs))
    print(f"{len(cpus)} CPUs, {ncores} physical cores in the affinity mask; "
          f"MKL_NUM_THREADS={threads}", flush=True)
    settings = {
        "default": {},
        "KMP_BLOCKTIME=infinite": {"KMP_BLOCKTIME": "infinite"},
        "OMP_PROC_BIND=close OMP_PLACES=cores": {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"},
    }
    for name, extra in settings.items():
        print(f"\n{name}", flush=True)
        for _ in range(4):
            env = dict(os.environ, MKL_NUM_THREADS=str(threads), **extra)
            subprocess.run([sys.executable, __file__, "--child"], env=env, check=True)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
