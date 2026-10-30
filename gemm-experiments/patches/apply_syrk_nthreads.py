"""Run syrk on fewer threads instead of one when n is small for the team.

usage: python apply_syrk_nthreads.py OPENBLAS_SRC

driver/level3/level3_syrk_threaded.c falls back to the single-threaded
kernel when n < nthreads * switch_ratio. With switch_ratio 16 (the x86-64
AVX-512 targets), X.T @ X with 990 features runs at 980 GF/s on 60 threads
of a Xeon 6787P socket and at 81 GF/s (one core) from 62 threads up. GEMM's
driver lowers its thread count in that case; do the same here.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "driver" / "level3" / "level3_syrk_threaded.c"
s = path.read_text()
old = "  if ((nthreads == 1) || (args->n < nthreads * switch_ratio)) {"
assert s.count(old) == 1
s = s.replace(old, """  if (args->n < nthreads * switch_ratio) nthreads = args->n / switch_ratio;
  if (nthreads <= 1) {""")
path.write_text(s)
print("patched", path)
