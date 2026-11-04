"""Compile probes/clauses_msvc.c with MSVC on Compiler Explorer, list the runtime calls per function.

usage: python -I msvc_godbolt.py probes/clauses_msvc.c COMPILER_ID...
"""
import json, re, sys, urllib.request

src = open(sys.argv[1]).read()
for cid in sys.argv[2:]:
    body = json.dumps({"source": src, "lang": "c", "options": {"userArguments": "/O2 /openmp",
        "filters": {"intel": True, "directives": True, "labels": True, "commentOnly": True}}}).encode()
    req = urllib.request.Request(f"https://godbolt.org/api/compiler/{cid}/compile", data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "prange-barrier-probe"})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    print("==", cid, "exit", d.get("code"))
    for l in d.get("stdout", []):
        if "error" in l.get("text", ""):
            print("  ", l["text"])
    fn, seen = None, {}
    for l in (x.get("text", "") for x in d.get("asm", [])):
        m = re.match(r"^\s*([$\w@?]+)\s+PROC", l) or re.match(r"^([$\w@?]+):\s*$", l)
        if m and not m.group(1).startswith("$"):
            fn = m.group(1); seen.setdefault(fn, [])
        m = re.search(r"\b(?:call|jmp)\s+(?:QWORD PTR\s+)?(?:__imp_)?(_?vcomp\w+|work)", l)
        if m and fn:
            seen[fn].append(m.group(1))
    for k, v in seen.items():
        print(f"  {k}: {' '.join(v)}")
