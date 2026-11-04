// Counts, on the main thread, GOMP_barrier calls by call site and GOMP_parallel calls by outlined
// function, as offsets in their module. Preload with libgomp.so, since RTLD_NEXT can't see a
// libgomp loaded by Python as a local dependency.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int omp_get_thread_num(void);
#define N 256
static struct { void *addr; long calls; char kind; } sites[N];

static void add(void *addr, char kind) {
    for (int i = 0; i < N; i++) {
        if (sites[i].addr == addr) { sites[i].calls++; return; }
        if (!sites[i].addr) { sites[i].addr = addr; sites[i].kind = kind; sites[i].calls = 1; return; }
    }
}

void GOMP_barrier(void) {
    static void (*real)(void);
    if (!real) real = dlsym(RTLD_NEXT, "GOMP_barrier");
    if (omp_get_thread_num() == 0) add(__builtin_return_address(0), 'b');
    real();
}

void GOMP_parallel(void (*fn)(void *), void *data, unsigned n, unsigned flags) {
    static void (*real)(void (*)(void *), void *, unsigned, unsigned);
    if (!real) real = dlsym(RTLD_NEXT, "GOMP_parallel");
    add((void *)fn, 'p');
    real(fn, data, n, flags);
}

__attribute__((destructor)) static void report(void) {
    for (int i = 0; i < N && sites[i].addr; i++) {
        Dl_info info;
        if (dladdr(sites[i].addr, &info) && info.dli_fname) {
            fprintf(stderr, "COUNT %c %s %lx %ld\n", sites[i].kind, info.dli_fname,
                    (unsigned long)((char *)sites[i].addr - (char *)info.dli_fbase), sites[i].calls);
        }
    }
}
