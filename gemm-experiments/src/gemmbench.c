// Times cblas_{s,d}gemm from a BLAS library loaded at runtime with dlopen, so
// the same binary compares OpenBLAS, MKL, BLIS, Accelerate and custom builds.
//
// usage: gemmbench --lib PATH [--prefix P] [--dtype d|s] [--min-time S]
//                  [--max-reps N] [--label L] [--init serial|parallel] SHAPE...
// SHAPE: MxNxK[:TT] with TT in {NN,NT,TN,TT} (column-major, C = op(A) op(B)).
// numpy's row-major X @ Y with X (m, k) and Y (k, n) is the column-major call
// Y^T X^T, i.e. shape NxMxK:NN here.
// Threads come from the environment (OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
// BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, OMP_NUM_THREADS); --threads N also
// calls the library's setter when it has one.
// Output: one CSV row per shape on stdout.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ColMajor = 102 };
enum { NoTrans = 111, Trans = 112 };

typedef void (*dgemm_t)(int, int, int, int, int, int, double, const double *,
                        int, const double *, int, double, double *, int);
typedef void (*sgemm_t)(int, int, int, int, int, int, float, const float *,
                        int, const float *, int, float, float *, int);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void *sym(void *h, const char *prefix, const char *name) {
    char buf[256];
    snprintf(buf, sizeof buf, "%s%s", prefix, name);
    return dlsym(h, buf);
}

struct fill_arg { char *p; size_t nbytes; int esize; unsigned seed; };

static void *fill(void *varg) {
    struct fill_arg *a = varg;
    unsigned s = a->seed;
    size_t n = a->nbytes / a->esize;
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        double v = (double)(s >> 8) / (1u << 24) - 0.5;
        if (a->esize == 8) ((double *)a->p)[i] = v;
        else ((float *)a->p)[i] = (float)v;
    }
    return NULL;
}

// Parallel init spreads first-touch pages over the threads' NUMA nodes.
static void *alloc_init(size_t nelem, int esize, int nthreads, unsigned seed) {
    size_t nbytes = (nelem * esize + 4095) / 4096 * 4096;
    char *p = aligned_alloc(4096, nbytes ? nbytes : 4096);
    if (!p) { perror("aligned_alloc"); exit(1); }
    if (nthreads < 1) nthreads = 1;
    pthread_t th[1024];
    struct fill_arg args[1024];
    if (nthreads > 1024) nthreads = 1024;
    size_t chunk = (nbytes / nthreads + 4095) / 4096 * 4096;
    int used = 0;
    for (int t = 0; t < nthreads; t++) {
        size_t off = t * chunk;
        if (off >= nbytes) break;
        size_t len = off + chunk > nbytes ? nbytes - off : chunk;
        args[t] = (struct fill_arg){p + off, len, esize, seed + 7919u * t};
        pthread_create(&th[t], NULL, fill, &args[t]);
        used++;
    }
    for (int t = 0; t < used; t++) pthread_join(th[t], NULL);
    return p;
}

int main(int argc, char **argv) {
    const char *lib = NULL, *prefix = "", *label = "", *init = "serial";
    char dtype = 'd';
    double min_time = 0.5;
    int max_reps = 100000, threads = 0, header = 0;
    int i = 1;
    for (; i < argc && strncmp(argv[i], "--", 2) == 0; i += 2) {
        if (!strcmp(argv[i], "--header")) { header = 1; i--; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
        const char *v = argv[i + 1];
        if (!strcmp(argv[i], "--lib")) lib = v;
        else if (!strcmp(argv[i], "--prefix")) prefix = v;
        else if (!strcmp(argv[i], "--dtype")) dtype = v[0];
        else if (!strcmp(argv[i], "--min-time")) min_time = atof(v);
        else if (!strcmp(argv[i], "--max-reps")) max_reps = atoi(v);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(v);
        else if (!strcmp(argv[i], "--label")) label = v;
        else if (!strcmp(argv[i], "--init")) init = v;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!lib) { fprintf(stderr, "--lib is required\n"); return 2; }
    void *h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    if (threads > 0) {
        void (*f)(int);
        if ((f = sym(h, prefix, "openblas_set_num_threads"))) f(threads);
        if ((f = dlsym(h, "MKL_Set_Num_Threads"))) f(threads);
        if ((f = dlsym(h, "bli_thread_set_num_threads"))) f(threads);
    }
    int (*get_threads)(void) = sym(h, prefix, "openblas_get_num_threads");
    if (!get_threads) get_threads = dlsym(h, "MKL_Get_Max_Threads");
    char *(*cfg)(void) = sym(h, prefix, "openblas_get_config");
    char *(*core)(void) = sym(h, prefix, "openblas_get_corename");
    if (cfg) fprintf(stderr, "# openblas config: %s core: %s\n", cfg(), core ? core() : "?");
    void (*mklver)(char *, int) = dlsym(h, "MKL_Get_Version_String");
    if (mklver) { char b[256]; mklver(b, sizeof b); fprintf(stderr, "# %s\n", b); }
    const char *(*bliver)(void) = dlsym(h, "bli_info_get_version_str");
    if (bliver) fprintf(stderr, "# blis %s\n", bliver());
    int nthr = get_threads ? get_threads() : threads;
    fprintf(stderr, "# library threads: %d\n", nthr);

    dgemm_t dgemm = sym(h, prefix, "cblas_dgemm");
    sgemm_t sgemm = sym(h, prefix, "cblas_sgemm");
    if (!dgemm || !sgemm) { fprintf(stderr, "cblas_?gemm not found\n"); return 1; }

    if (header)
        printf("label,dtype,m,n,k,trans,threads,reps,best_s,median_s,"
               "gflops_best,gflops_median\n");
    int init_threads = !strcmp(init, "parallel") ? (nthr > 0 ? nthr : 1) : 1;
    int esize = dtype == 'd' ? 8 : 4;
    for (; i < argc; i++) {
        long m, n, k;
        char tt[3] = "NN";
        if (sscanf(argv[i], "%ldx%ldx%ld:%2s", &m, &n, &k, tt) < 3) {
            fprintf(stderr, "bad shape %s\n", argv[i]);
            return 2;
        }
        int ta = tt[0] == 'T' ? Trans : NoTrans, tb = tt[1] == 'T' ? Trans : NoTrans;
        long lda = ta == NoTrans ? m : k, ldb = tb == NoTrans ? k : n, ldc = m;
        void *A = alloc_init((size_t)m * k, esize, init_threads, 1);
        void *B = alloc_init((size_t)k * n, esize, init_threads, 2);
        void *C = alloc_init((size_t)m * n, esize, init_threads, 3);
        double flops = 2.0 * m * n * k;
#define CALL()                                                                 \
    do {                                                                       \
        if (dtype == 'd')                                                      \
            dgemm(ColMajor, ta, tb, m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc); \
        else                                                                   \
            sgemm(ColMajor, ta, tb, m, n, k, 1.0f, A, lda, B, ldb, 0.0f, C, ldc); \
    } while (0)
        // Warm up the thread pool, caches and page tables for at least 50 ms.
        double t0 = now();
        int w = 0;
        do { CALL(); w++; } while (now() - t0 < 0.05 && w < max_reps);
        int cap = 1 << 16;
        double *times = malloc(cap * sizeof(double));
        int reps = 0;
        double start = now();
        while (reps < max_reps && (reps < 3 || now() - start < min_time)) {
            double a = now();
            CALL();
            double b = now();
            if (reps == cap) { cap *= 2; times = realloc(times, cap * sizeof(double)); }
            times[reps++] = b - a;
        }
        qsort(times, reps, sizeof(double), cmp_double);
        double best = times[0], med = times[reps / 2];
        printf("%s,%c,%ld,%ld,%ld,%s,%d,%d,%.9g,%.9g,%.4f,%.4f\n", label, dtype,
               m, n, k, tt, nthr, reps, best, med, flops / best * 1e-9,
               flops / med * 1e-9);
        fflush(stdout);
        free(times);
        free(A); free(B); free(C);
    }
    return 0;
}
