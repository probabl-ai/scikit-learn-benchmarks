# cython: language_level=3, freethreading_compatible=True
from cython.parallel import prange
def target_read_after(int n=1000, int nthreads=4):
    """Holds the GIL: the target stays lastprivate since it's read after the loop."""
    cdef int i = -1, total = 0
    for i in prange(n, num_threads=nthreads):
        total += 1
    return i, total
def target_unused(int n=1000, int nthreads=4):
    cdef int i, total = 0
    for i in prange(n, num_threads=nthreads):
        total += 1
    return total
