"""Reuse a small packed-B slot when no other thread reads this thread's B.

usage: python apply_l1stride.py OPENBLAS_SRC

In driver/level3/level3_thread.c, inner_thread packs its share of B
(div_n = range/DIVIDE_RATE columns, often several MB) into a buffer that the
other threads of its column group then read. With nthreads_m == 1 nobody
else reads it, and when m fits in one GEMM_P block the buffer is not reused
for later m blocks either, so each packed micro-panel is consumed right away,
exactly like the single-threaded case, which already sets l1stride = 0.
Keeping l1stride = 1 there streams the whole packed copy of B through memory.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "driver" / "level3" / "level3_thread.c"
s = path.read_text()
old = "\tif (args -> nthreads == 1) l1stride = 0;"
assert s.count(old) == 1
s = s.replace(old, "\tif (args -> nthreads == 1 || nthreads_m == 1) l1stride = 0;")
path.write_text(s)
print("patched", path)
