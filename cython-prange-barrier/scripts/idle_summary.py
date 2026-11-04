"""Idle state entries and time per fit, P-cores and E-cores apart, from idle_probe.sh snapshots.

usage: idle_summary.py DIR P_LAST   (P-cores are cpu0 to cpuP_LAST)
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r"cpu(\d+)/cpuidle/state(\d+)/(\w+):(.*)")


def load(path):
    values, names = {}, {}
    for line in path.read_text().splitlines():
        m = PAT.search(line)
        cpu, state, field, value = int(m[1]), int(m[2]), m[3], m[4]
        if field == "name":
            names[state] = value
        else:
            values[(cpu, state, field)] = int(value)
    return values, names


p_last = int(sys.argv[2])
usage = defaultdict(list)
times = defaultdict(list)
for after in sorted(Path(sys.argv[1]).glob("*.after")):
    cfg, build, _ = after.stem.rsplit("-", 2)
    before, names = load(after.with_suffix(".before"))
    now, _ = load(after)
    tot = defaultdict(int)
    for (cpu, state, field), value in now.items():
        tot[("P" if cpu <= p_last else "E", state, field)] += value - before[(cpu, state, field)]
    for kind in "PE":
        for state, name in names.items():
            usage[(cfg, build, kind, name)].append(tot[(kind, state, "usage")])
            times[(cfg, build, kind, name)].append(tot[(kind, state, "time")] / 1e6)

print("| cfg | build | cores | state | entries per fit | idle s per fit (sum over cores) |")
print("| --- | --- | --- | --- | --- | --- |")
for key in sorted(usage):
    print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {statistics.median(usage[key]):.0f} | {statistics.median(times[key]):.1f} |")
