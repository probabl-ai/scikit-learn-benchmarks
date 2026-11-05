"""Exps 22, 25, 26: fit time at the top thread counts, every case above a size, sorted by size.
Ratios are against the lowest column. usage: size_table.py MIN_SIZE COL,COL RESULTS...
A column is the thread count used, read from the phases."""
import json, re, statistics as st, sys
from collections import defaultdict

SHAPES = {"L": (100_000, 100, 31), "L-thin": (500_000, 10, 31), "L-stumps": (1_000_000, 100, 2), "L-leaf255": (100_000, 100, 255),
          "susy": (5_000_000 * 4 // 5, 18, 31), "year_prediction_msd": (463_715, 90, 31), "covtype": (464_810, 54, 31), "kddcup09_churn": (40_000, 230, 31)}


def shape(c, wl):
    if c in SHAPES:
        return SHAPES[c]
    return wl.get(c)


minsize, cols, paths = float(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
rows = defaultdict(dict)
WORK = {}
for src in ("run_exp25.sh", "run_exp26.sh", "run_exp22.sh"):
    for name, n, f, l in re.findall(r"([\w.-]+):(\d+):(\d+)(?::(\d+))?(?=[,\s])", open(f"scripts/ab/{src}").read()):
        WORK[name] = (int(n), int(f), int(l or 31))
for path in paths:
    fits = defaultdict(list)
    prof = path.split("/")[-1].removesuffix(".jsonl").split("-", 1)[1]
    for line in open(path):
        r = json.loads(line)
        if r["rc"] == 0 and r["fit_s"] and "c325main" not in r["variant"] and "~FS" not in r["variant"]:
            fits[r["case"], r["phases"]["n_threads"]].append(st.median(r["fit_s"]))
    for (c, t), v in fits.items():
        rows[c].setdefault(prof, {})[t] = st.median(v)
key = lambda c: (shape(c, WORK) or (0, 0, 0))
cases = sorted((c for c in rows if key(c)[0] * key(c)[1] >= minsize), key=lambda c: key(c)[0] * key(c)[1])
profs = list(dict.fromkeys(p for c in cases for p in rows[c]))
print("| case | size | leaves | " + " | ".join(profs) + " |")
print("|" + " --- |" * (3 + len(profs)))
for c in cases:
    n, f, l = key(c)
    cells = []
    for p in profs:
        d = rows[c].get(p, {})
        pick = {t: d[t] for t in cols if t in d}
        # closest thread counts around each wanted column, within 25%
        for t in cols:
            if t not in pick:
                near = [x for x in d if abs(x - t) <= 0.1 * t]
                if near:
                    pick[t] = d[min(near, key=lambda x: abs(x - t))]
        if len(pick) == len(cols):
            base = pick[cols[0]]
            cells.append(" ".join(f"{pick[t] / base:.2f}" for t in cols[1:]))
        else:
            cells.append("")
    print(f"| {c} | {n * f / 1e6:.0f}M | {l} | " + " | ".join(cells) + " |")
