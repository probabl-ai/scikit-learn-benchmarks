# cython: language_level=3, freethreading_compatible=True
from cython.parallel import prange

def run(int n=1000):
    cdef int i
    cdef int total = 0
    for i in prange(n, num_threads=4):  # no nogil=True: the GIL stays held
        total += 1
    return total
