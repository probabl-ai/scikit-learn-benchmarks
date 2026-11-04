# cython: language_level=3, boundscheck=False, wraparound=False
# Does the code after a prange in a `with parallel` block see all of its writes?
from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport calloc, free
from posix.unistd cimport usleep

def after_loop(int n, int nthreads):
    """Number of threads that, right after the prange, still see unwritten items."""
    cdef int i, k, seen_missing = 0
    cdef int *a = <int *> calloc(n, sizeof(int))
    cdef int *counter = &seen_missing
    with nogil, parallel(num_threads=nthreads):
        for i in prange(n, schedule='static'):
            if i == n - n // nthreads:
                usleep(20000)  # the thread owning the last chunk starts late
            a[i] = 1
        # In OpenMP terms: a worksharing loop without nowait ends with a barrier
        for k in range(n):
            if a[k] == 0:
                with gil:
                    counter[0] += 1
                break
    free(a)
    return seen_missing

def between_loops(int n, int nthreads):
    """Items that the second prange found unwritten by the first one."""
    cdef int i, j, missing = 0
    cdef int *a = <int *> calloc(n, sizeof(int))
    with nogil, parallel(num_threads=nthreads):
        for i in prange(n, schedule='static'):
            if i == n - n // nthreads:
                usleep(20000)
            a[i] = 1
        for j in prange(n, schedule='static'):
            if a[n - 1 - j] == 0:
                missing += 1
    free(a)
    return missing
