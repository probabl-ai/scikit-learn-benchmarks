"""Run numpy eigh in a loop and sample which CPU each thread of this process is on.

Prints per sample: eigh time and the CPUs that hold more than one thread.
"""

import collections
import os
import sys
import threading
import time

import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(0)
A = rng.standard_normal((n, n))
S = A + A.T
times = []
stop = False


def work():
    while not stop:
        t = time.perf_counter()
        np.linalg.eigh(S)
        times.append(time.perf_counter() - t)


def cpus():
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            stat = open(f"/proc/self/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            out[int(tid)] = (stat[0], int(stat[36]))  # state, processor
        except OSError:
            pass
    return out


th = threading.Thread(target=work)
th.start()
me = threading.get_native_id()
for _ in range(int(os.environ.get("SAMPLES", 8))):
    time.sleep(1)
    snap = {t: v for t, v in cpus().items() if t != me}
    running = collections.Counter(cpu for state, cpu in snap.values() if state == "R")
    allc = collections.Counter(cpu for state, cpu in snap.values())
    shared = {c: k for c, k in allc.items() if k > 1}
    last = times[-1] if times else float("nan")
    print(f"eigh {last * 1e3:8.1f} ms  threads={len(snap)} running={sum(running.values())} "
          f"cpus_used={len(allc)} shared={shared}", flush=True)
stop = True
th.join()
