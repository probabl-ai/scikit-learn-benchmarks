"""sklbench's estimator runner, plus the grower's phase times of all the fits.

Writes one JSON object to $HGB_PHASE_OUT at exit: the summed per-phase times (s),
the number of grow() calls, and the machine's idle state entries since startup.
"""
import atexit, glob, json, os, runpy, sys

from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower



def idle_counts():
    counts = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpuidle/state*/usage"):
        state = path.split("/")[-2]
        counts[state] = counts.get(state, 0) + int(open(path).read())
    return counts


idle_start = idle_counts()
totals = {"hist": 0.0, "find": 0.0, "apply": 0.0, "grow": 0.0, "n_nodes": 0, "n_trees": 0}
grow = TreeGrower.grow


def timed_grow(self):
    from time import perf_counter
    t0 = perf_counter()
    grow(self)
    totals["grow"] += perf_counter() - t0
    totals["hist"] += self.total_compute_hist_time
    totals["find"] += self.total_find_split_time
    totals["apply"] += self.total_apply_split_time
    totals["n_nodes"] += self.n_nodes
    totals["n_trees"] += 1
    totals["n_threads"] = self.splitter.n_threads
    totals["active_wait"] = bool(getattr(self.splitter, "active_wait", False))


TreeGrower.grow = timed_grow


def dump():
    end = idle_counts()
    totals["idle_entries"] = {k: end[k] - idle_start[k] for k in end}
    json.dump(totals, open(os.environ["HGB_PHASE_OUT"], "w"))


if os.environ.get("HGB_PHASE_OUT"):
    atexit.register(dump)
sys.argv = ["sklbench.runners.estimator"] + sys.argv[1:]
runpy.run_module("sklbench.runners.estimator", run_name="__main__", alter_sys=True)
