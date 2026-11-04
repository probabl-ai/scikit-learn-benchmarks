# cython: language_level=3
from cython.parallel import parallel, prange
from libc.stdlib cimport calloc, free
from posix.unistd cimport usleep

def run(int n=1000, int nthreads=4):
    cdef int i, k, late = n - n // nthreads, n_stale = 0
    cdef int *n_stale_ptr = &n_stale
    cdef int *a = <int *> calloc(n, sizeof(int))
    with nogil, parallel(num_threads=nthreads):
        for i in prange(n, schedule='static'):
            if i == late:
                usleep(20000)  # the thread with the last chunk starts late
            a[i] = 1
        # Every iteration of the prange should be finished here
        for k in range(n):
            if a[k] == 0:
                with gil:
                    n_stale_ptr[0] += 1
                break
    free(a)
    return n_stale  # number of threads that saw an unwritten item
