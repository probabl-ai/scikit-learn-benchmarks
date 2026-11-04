"""For each prange in a `with parallel` block, print the code that follows it in that block.

That code runs without an end barrier with Cython 3.3 and GCC. usage: audit_parallel_blocks.py SRC_DIR
"""
import re
import sys
from pathlib import Path


def indent(line):
    return len(line) - len(line.lstrip())


for path in sorted(Path(sys.argv[1]).rglob("*.pyx*")):
    lines = path.read_text().split("\n")
    for i, line in enumerate(lines):
        if not re.search(r"with .*\bparallel\(", line) or line.strip().startswith("#"):
            continue
        block = indent(line)
        end = i + 1
        while end < len(lines) and (not lines[end].strip() or indent(lines[end]) > block):
            end += 1
        for j in range(i + 1, end):
            if "prange(" not in lines[j] or lines[j].strip().startswith("#"):
                continue
            loop = indent(lines[j])
            # Skip the arguments on continuation lines, then the body of the loop.
            k = j
            while not lines[k].rstrip().endswith(":"):
                k += 1
            k += 1
            while k < end and (not lines[k].strip() or indent(lines[k]) > loop):
                k += 1
            after = [x for x in lines[k:end] if x.strip() and not x.strip().startswith("#")]
            print(f"\n== {path.relative_to(sys.argv[1])}:{j + 1}: {len(after)} lines after the loop")
            for x in after[:14]:
                print("   ", x.strip()[:100])
