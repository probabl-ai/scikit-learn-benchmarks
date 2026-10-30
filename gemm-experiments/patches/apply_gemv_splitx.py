"""Use the column-split (split-x) gemv_n path when rows per thread are few.

usage: python apply_gemv_splitx.py OPENBLAS_SRC

driver/level2/gemv_thread.c splits y = A x (A m x n, column-major) over the m
rows, and only falls back to splitting the n columns (with a per-thread
partial y and a reduction) when it cannot make nthreads row blocks, and only
if m * nthreads <= 1024 (a static thread-local buffer). For X.T @ r in
scikit-learn's linear models (m = n_features ~ 100, n = n_samples) that
gives each thread a 4-16 row strip of every column: a short, strided kernel
call that barely scales. Prefer split-x when the row strips are under 64
rows, and allocate the partial-y buffer when the static one is too small.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "driver" / "level2" / "gemv_thread.c"
s = path.read_text()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


sub("""#if !defined(TRANSA) && !defined(UNSAFE)
  int split_x=0;
#endif""", """#if !defined(TRANSA) && !defined(UNSAFE)
  int split_x=0;
  FLOAT *ybuf = y_dummy;
#endif""")
sub("""  if (num_cpu < nthreads) {
""", """  if (num_cpu < nthreads || range[1] - range[0] < 64) {
""")
sub("""	 || m*COMPSIZE*nthreads > Y_DUMMY_NUM)
      goto Outer;
""", """	 || n < 16 * nthreads)
      goto Outer;
    if (m*COMPSIZE*nthreads > Y_DUMMY_NUM) {
      ybuf = (FLOAT *)malloc(sizeof(FLOAT) * m * COMPSIZE * nthreads);
      if (ybuf == NULL) { ybuf = y_dummy; goto Outer; }
    }
""")
sub("memset(y_dummy, 0, sizeof(FLOAT) * m * COMPSIZE * nthreads);",
    "memset(ybuf, 0, sizeof(FLOAT) * m * COMPSIZE * nthreads);")
sub("args.c = (void *)y_dummy;", "args.c = (void *)ybuf;")
sub("y[j*incy*COMPSIZE] +=y_dummy[i*m*COMPSIZE + j*COMPSIZE];",
    "y[j*incy*COMPSIZE] +=ybuf[i*m*COMPSIZE + j*COMPSIZE];")
sub("y[j*incy*COMPSIZE+1] +=y_dummy[i*m*COMPSIZE + j*COMPSIZE+1];",
    "y[j*incy*COMPSIZE+1] +=ybuf[i*m*COMPSIZE + j*COMPSIZE+1];")
sub("""      }
    }
  }
#endif

  return 0;
}""", """      }
    }
    if (ybuf != y_dummy) free(ybuf);
  }
#endif

  return 0;
}""")
path.write_text(s)
print("patched", path)
