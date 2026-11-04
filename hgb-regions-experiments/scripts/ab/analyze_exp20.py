"""Exp 20: year_prediction_msd in seconds, and the effect of the Cython version on the same source."""
import json
import statistics as st
import sys
from collections import defaultdict


def load(path):
    fits = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        fits[r["case"], r["variant"]].append(st.median(r["fit_s"]))
    return fits


print("| profile | main 3.2.5 | main 3.3 | #34935 3.2.5 | #34935 3.3* | PR 3.2.5 | PR 3.3 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for path in sys.argv[1:]:
    a, m = load(path), load(path.removesuffix(".jsonl") + "-main.jsonl")
    c = "year_prediction_msd"
    cells = [m[c, "c325main:main"], m[c, "main"], a[c, "c325pr:pr"], a[c, "pr"], a[c, "c325clean:pr"], a[c, "clean:pr"]]
    name = path.split("/")[-1].removesuffix(".jsonl").removeprefix("exp20-")
    print(f"| {name} | " + " | ".join(f"{st.median(x):.3g}" for x in cells) + " |")

print("\n| profile | geomean PR/#34935 3.2.5 | geomean PR/#34935 3.3* | PR 3.3/3.2.5 | #34935 3.3*/3.2.5 | worst case PR/#34935 3.2.5 |")
print("| --- | --- | --- | --- | --- | --- |")
for path in sys.argv[1:]:
    a = load(path)
    cases = list(dict.fromkeys(k[0] for k in a))
    gm = lambda x, y: st.geometric_mean(st.median(a[k, x]) / st.median(a[k, y]) for k in cases)
    worst = max(cases, key=lambda k: st.median(a[k, "c325clean:pr"]) / st.median(a[k, "c325pr:pr"]))
    w = st.median(a[worst, "c325clean:pr"]) / st.median(a[worst, "c325pr:pr"])
    name = path.split("/")[-1].removesuffix(".jsonl").removeprefix("exp20-")
    print(f"| {name} | {gm('c325clean:pr', 'c325pr:pr'):.2f} | {gm('clean:pr', 'pr'):.2f} | {gm('clean:pr', 'c325clean:pr'):.2f} | {gm('pr', 'c325pr:pr'):.2f} | {worst} {w:.2f} |")
