# cython: language_level=3, boundscheck=False, wraparound=False
# What each kind of prange generates, and which warnings its clauses silence.
from cython.parallel import prange, parallel

def target_unused(int n, double[:] a):
    cdef Py_ssize_t i
    for i in prange(n, nogil=True, schedule='static'):
        a[i] = i

def target_read_after(int n, double[:] a):
    cdef Py_ssize_t i = -1
    for i in prange(n, nogil=True, schedule='static'):
        a[i] = i
    return i

def private_read_after(int n, double[:] a):
    cdef Py_ssize_t i
    cdef double x = 0
    for i in prange(n, nogil=True, schedule='static'):
        x = a[i] * 2
        a[i] = x
    return x

def conditional_private(int n, double[:] a):
    cdef Py_ssize_t i
    cdef double x = 7
    for i in prange(n, nogil=True, schedule='static'):
        if a[i] > 0:
            x = a[i]
    return x

def read_before_write(int n, double[:] a):
    cdef Py_ssize_t i
    cdef double x = 7
    for i in prange(n, nogil=True, schedule='static'):
        a[i] = x
        x = i
    return x

def reduction(int n, double[:] a):
    cdef Py_ssize_t i
    cdef double s = 0
    for i in prange(n, nogil=True, schedule='static'):
        s += a[i]
    return s

def with_break(int n, double[:] a):
    cdef Py_ssize_t i
    cdef double x = 7
    for i in prange(n, nogil=True, schedule='static'):
        x = a[i]
        if x < 0:
            break
    return i, x

def with_parallel_two(int n, double[:] a, double[:] b):
    cdef Py_ssize_t i, j
    with nogil, parallel():
        for i in prange(n, schedule='static'):
            a[i] = i
        for j in prange(n, schedule='static'):
            b[j] = a[n - 1 - j]

def with_parallel_tmp(int n, double[:] a):
    cdef Py_ssize_t i
    cdef double tmp
    with nogil, parallel():
        tmp = 1
        for i in prange(n, schedule='static'):
            a[i] = i * tmp
