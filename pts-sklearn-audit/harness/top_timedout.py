"""Where the timed-out runs spent their main-thread time: script line -> leaf frame."""
import json
from collections import Counter
from pathlib import Path

prof = Path("prof/xeon-s1")
for line in (prof / "runs.jsonl").read_text().splitlines():
    r = json.loads(line)
    if not r["timed_out"] or not (prof / r["raw"]).exists():
        continue
    main = "thread (%d)" % r["pid"]
    c, tot = Counter(), 0
    for l in (prof / r["raw"]).read_text(errors="replace").splitlines():
        st, _, n = l.rpartition(" ")
        if not n.isdigit():
            continue
        parts = st.split(";")
        if not parts[0].startswith(main):
            continue
        tot += int(n)
        own = [x for x in parts[1:] if "bench_" in x]
        c[(own[-1] if own else "-")[:70] + "  ->  " + parts[-1][:80]] += int(n)
    print("==", r["entry"], "samples", tot)
    for k, n in c.most_common(4):
        print(f"  {n / tot:5.0%} {k}")
