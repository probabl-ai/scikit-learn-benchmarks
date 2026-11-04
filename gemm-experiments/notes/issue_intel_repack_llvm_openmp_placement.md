Target: conda-forge/intel_repack-feedstock (cross-link #115)

Title: MKL is up to ~2x slower on many-core machines with llvm-openmp than with Intel's OpenMP runtime

---

On a many-core server, conda-forge's MKL runs matrix products up to ~2x slower than the same MKL with Intel's OpenMP runtime (libiomp5). Its speed also changes a lot from one process to the next. Since MKL on conda-forge always runs on llvm-openmp, every MKL user on such a machine gets the slow version by default.

Measured on one socket of a Xeon 6787P (86 cores, 172 hardware threads), MKL 2026.1, 86 threads, `cblas_dgemm` called directly, GFLOP/s (higher is better):

| | 4096^3 | 7x116203x100 | 100x100x1e6 (X.T @ Y) | 100x200000x100 |
| --- | ---: | ---: | ---: | ---: |
| conda-forge (llvm-openmp 23.1.4) | 2554 | 1283 | 1781 | 2665 |
| same MKL, Intel libiomp5 2026.1.2 preloaded | 4787 | 3021 | 3281 | 5024 |

- Through numpy, `a @ a` (4096x4096) is ~1.35x slower than with the threads bound to cores.
- With 43 threads, the same 4096^3 product runs at 1436 or 3227 GFLOP/s depending on the process. Benchmarks with MKL on this machine are hard to trust because of it.
- It starts around 43 threads on this socket and gets worse with more. I haven't checked other CPUs.

The reproducer uses only numpy and MKL from conda-forge (Linux). It runs `a @ a` for 3 s in fresh processes, with one MKL thread per physical core, and reports GFLOP/s and how many distinct physical cores the threads run on:

```
pixi exec -s python=3.12 -s numpy -s "libblas=*=*mkl" -- python repro_libomp_placement.py
```

<details><summary>repro_libomp_placement.py</summary>

```python
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
```

</details>

Output on one socket (`numactl --cpunodebind --membind` to that socket, nothing else running on it), abbreviated:

```
packages: libblas-3.11.0-11_h5875eb1_mkl, llvm-openmp-23.1.4-h7148c6a_0, mkl-2026.1.0-hd2095e1_246, numpy-2.5.4-py312he827f4e_0
172 CPUs, 86 physical cores in the affinity mask; MKL_NUM_THREADS=86

default
    2232 GFLOP/s   running  86.0   cpus  86.0   cores  63.7
    2291 GFLOP/s   running  86.0   cpus  86.0   cores  65.3
    2440 GFLOP/s   running  86.0   cpus  84.9   cores  52.9

OMP_PROC_BIND=close OMP_PLACES=cores
    3133 GFLOP/s   running  86.0   cpus  86.0   cores  86.0
    3047 GFLOP/s   running  86.0   cpus  86.0   cores  86.0
    3073 GFLOP/s   running  86.0   cpus  86.0   cores  86.0
```

Expected: the 86 threads on 86 cores, at the speed MKL gets with Intel's runtime. Actual: by default only ~53-67 distinct cores are busy, so 20-33 threads share a core with another one.

Workarounds today: preload `libiomp5.so` from the pip `intel-openmp` wheel (conda-forge has no `intel-openmp` package for linux-64), or set `KMP_BLOCKTIME=infinite` (then idle threads spin forever, which slows down any other thread pool in the process).

I can run more tests on this machine if that helps. Full data: https://github.com/probabl-ai/scikit-learn-benchmarks/blob/gemm-experiments/gemm-experiments/FINDINGS.md (section 2).

---

Not for the first post. Possible follow-up comment if maintainers ask for details:

- `KMP_SETTINGS=1` reports the same settings for both runtimes (`KMP_BLOCKTIME=200ms`, `KMP_LIBRARY=throughput`, hyper barriers), so #115's "identical at runtime" holds for settings but not for placement. Intel's runtime also reports `KMP_AFFINITY=...,nobind_init,perf_bind`, which llvm-openmp doesn't.
- The threads don't sleep (< 0.2 voluntary context switches per thread per call) or migrate more than with libiomp5, and `strace -c` shows similar `sched_yield` and `futex` counts.
- At 172 threads, 35-64 threads are stacked two per CPU (millions of involuntary context switches).
- GNU libgomp (`MKL_THREADING_LAYER=GNU`) behaves like llvm-openmp.
- Binding is not a good default: `close` costs 1.6-1.9x on memory-bound calls vs libiomp5 unbound, and two bound processes pile on the same cores. threadpoolctl can't set it either (ignored after the first `mkl_get_max_threads` call).
