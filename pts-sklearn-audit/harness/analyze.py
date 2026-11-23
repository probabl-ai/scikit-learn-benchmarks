"""Split each PTS scikit-learn score into time inside scikit-learn and the rest.

Usage: python analyze.py WORK_DIR RUN_ID

Reads the py-spy raw profiles and runs.jsonl written by shim/python3, and the
PTS result files, and writes WORK_DIR/analysis-RUN_ID.{json,md}.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

RATE = 50  # py-spy --rate in shim/python3
FRAME = re.compile(r"^(?P<func>.*) \((?P<file>[^()]*?)(?::(?P<line>\d+))?\)$")
CATEGORIES = ["sklearn", "sklearn.datasets", "import", "matplotlib", "other", "unsampled"]


def parse_frame(frame):
    m = FRAME.match(frame)
    return (m["func"], m["file"]) if m else (frame, "")


def category(frames):
    files = [f for _, f in frames]
    funcs = [fn for fn, _ in frames]
    if any("importlib" in f for f in files) or "_find_and_load" in funcs:
        return "import"
    if any("sklearn/datasets/" in f for f in files):
        return "sklearn.datasets"
    if any("sklearn/" in f for f in files):
        return "sklearn"
    if any("matplotlib/" in f or "mpl_toolkits/" in f for f in files):
        return "matplotlib"
    return "other"


def callee(frames):
    """What the benchmark script was calling: the frame after its last own frame."""
    last_script = max(
        (i for i, (_, f) in enumerate(frames) if Path(f).name.startswith("bench_")),
        default=None,
    )
    if last_script is None:
        return "(outside the script)"
    if last_script + 1 >= len(frames):
        fn, f = frames[last_script]
        return f"{Path(f).name}:{fn} (own code)"
    fn, f = frames[last_script + 1]
    mod = f.split("site-packages/")[-1] if "site-packages/" in f else Path(f).name
    return f"{mod}:{fn}"


def parse_raw(path, pid):
    """Main-thread sample counts per category, and outside-sklearn callees."""
    cats, callees = Counter(), Counter()
    main = f"thread ({pid})"
    for line in path.read_text(errors="replace").splitlines():
        stack, _, count = line.rpartition(" ")
        if not count.isdigit():
            continue
        parts = stack.split(";")
        if parts[0] != main and not parts[0].startswith(main + ":"):
            continue  # worker threads: their time is covered by the main thread's call
        frames = [parse_frame(p) for p in parts[1:]]
        c = category(frames)
        cats[c] += int(count)
        if c != "sklearn":
            callees[(c, callee(frames))] += int(count)
    return cats, callees


def pts_results(home, run_id):
    """{benchmark option value: (average, [per-run values])} from the PTS result files."""
    out = {}
    for comp in sorted((home / ".phoronix-test-suite/test-results").glob(f"skl-{run_id}-*/composite.xml")):
        for res in ET.parse(comp).getroot().iter("Result"):
            value = (res.findtext("Description") or "").removeprefix("Benchmark: ")
            for entry in res.iter("Entry"):
                try:  # failed tests have an empty Value
                    avg = float(entry.findtext("Value"))
                except (TypeError, ValueError):
                    continue
                raw = entry.findtext("RawString") or ""
                out[value] = (avg, [float(v) for v in raw.split(":") if v])
    return out


def main(work, run_id):
    work = Path(work)
    prof = work / "prof" / run_id
    runs = [json.loads(line) for line in (prof / "runs.jsonl").read_text().splitlines()]
    entries = {}
    for line in (work / "entries.txt").read_text().splitlines():
        value, name = line.split("|")
        entries[value] = name
    pts = pts_results(work / "home", run_id)

    by_entry = defaultdict(list)
    for r in runs:
        by_entry[r["entry"].removeprefix("bench_")].append(r)

    rows = []
    for value, name in entries.items():
        rs = by_entry.get(value, [])
        # PTS ignores the first run (IgnoreRuns=1): it also absorbs downloads and caches.
        counted = [r for r in rs if r["run"] >= 1 and r["rc"] == 0]
        cats, callees = Counter(), Counter()
        for r in counted:
            c, k = parse_raw(prof / r["raw"], r["pid"])
            # Wall time without samples: interpreter start-up before py-spy attached.
            c["unsampled"] = max(0, round(r["wall_s"] * RATE) - sum(c.values()))
            cats += c
            callees += k
        total = sum(cats.values())
        pts_avg, pts_values = pts.get(value, (None, []))
        row = {
            "name": name,
            "value": value,
            "runs": len(rs),
            "failed_runs": sum(r["rc"] != 0 for r in rs),
            "timed_out": any(r["timed_out"] for r in rs),
            "pts_score_s": pts_avg,
            "pts_values_s": pts_values,
            "python_wall_s": (sum(r["wall_s"] for r in counted) / len(counted)) if counted else None,
            "share": {c: cats[c] / total for c in CATEGORIES} if total else None,
            "top_outside": [
                {"category": c, "callee": k, "share": n / total}
                for (c, k), n in callees.most_common(5)
            ] if total else [],
        }
        if total and pts_avg:
            row["sklearn_s"] = row["share"]["sklearn"] * row["python_wall_s"]
        rows.append(row)

    (work / f"analysis-{run_id}.json").write_text(json.dumps(rows, indent=1))

    lines = [
        "| Entry | PTS score (s) | in sklearn | sklearn.datasets | imports | matplotlib | other | unsampled | top outside sklearn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r["share"] is None:
            status = "timed out" if r["timed_out"] else ("failed" if r["failed_runs"] else "not run")
            lines.append(f"| {r['name']} | {status} | | | | | | | |")
            continue
        s = r["share"]
        top = "; ".join(
            f"{t['callee']} {t['share']:.0%}" for t in r["top_outside"][:2] if t["share"] >= 0.02
        )
        score = f"{r['pts_score_s']:.1f}" if r["pts_score_s"] else "n/a"
        lines.append(
            f"| {r['name']} | {score} | {s['sklearn']:.0%} | {s['sklearn.datasets']:.0%} | "
            f"{s['import']:.0%} | {s['matplotlib']:.0%} | {s['other']:.0%} | {s['unsampled']:.0%} | {top} |"
        )
    (work / f"analysis-{run_id}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
