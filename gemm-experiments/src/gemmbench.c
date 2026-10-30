// Times cblas_{s,d}gemm from a BLAS library loaded at runtime with dlopen, so
// the same binary compares OpenBLAS, MKL, BLIS, Accelerate and custom builds.
//
// usage: gemmbench --lib PATH [--prefix P] [--suffix S] [--dtype d|s] [--min-time S]
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
typedef void (*dsyrk_t)(int, int, int, int, int, double, const double *, int,
                        double, double *, int);
typedef void (*ssyrk_t)(int, int, int, int, int, float, const float *, int,
                        float, float *, int);
typedef void (*dgemv_t)(int, int, int, int, double, const double *, int,
                        const double *, int, double, double *, int);
typedef void (*sgemv_t)(int, int, int, int, float, const float *, int,
                        const float *, int, float, float *, int);
enum { Upper = 121 };

static dgemm_t dgemm;
static sgemm_t sgemm;
static dsyrk_t dsyrk;
static ssyrk_t ssyrk;
static dgemv_t dgemv;
static sgemv_t sgemv;

struct op {
    char routine;  // 'g' gemm, 's' syrk, 'v' gemv
    char dtype;
    long m, n, k;
    int ta, tb;
    double beta;
    void *A, *B, *C;
};

static double op_flops(const struct op *o) {
    if (o->routine == 'g') return 2.0 * o->m * o->n * o->k;
    if (o->routine == 's') return 1.0 * o->n * (o->n + 1) * o->k;
    return 2.0 * o->m * o->n;
}

static void op_call(const struct op *o) {
    int d = o->dtype == 'd';
    if (o->routine == 'g') {
        long lda = o->ta == NoTrans ? o->m : o->k;
        long ldb = o->tb == NoTrans ? o->k : o->n;
        if (d) dgemm(ColMajor, o->ta, o->tb, o->m, o->n, o->k, 1.0, o->A, lda, o->B, ldb, o->beta, o->C, o->m);
        else sgemm(ColMajor, o->ta, o->tb, o->m, o->n, o->k, 1.0f, o->A, lda, o->B, ldb, (float)o->beta, o->C, o->m);
    } else if (o->routine == 's') {
        long lda = o->ta == NoTrans ? o->n : o->k;
        if (d) dsyrk(ColMajor, Upper, o->ta, o->n, o->k, 1.0, o->A, lda, 0.0, o->C, o->n);
        else ssyrk(ColMajor, Upper, o->ta, o->n, o->k, 1.0f, o->A, lda, 0.0f, o->C, o->n);
    } else {
        if (d) dgemv(ColMajor, o->ta, o->m, o->n, 1.0, o->A, o->m, o->B, 1, 0.0, o->C, 1);
        else sgemv(ColMajor, o->ta, o->m, o->n, 1.0f, o->A, o->m, o->B, 1, 0.0f, o->C, 1);
    }
}

static double now(void) {
    struct timespec ts;
#ifdef __APPLE__
    // macOS's CLOCK_MONOTONIC only has microsecond resolution.
    clock_gettime(CLOCK_UPTIME_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
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

// Compare one gemm call against a naive double-precision reference, on a copy
// of C so that beta != 0 is checked too. Returns max abs error / max abs ref.
// The inputs have 24-bit mantissas, so float64 results are often exact.
static double check_gemm(struct op *o, int esize) {
    long m = o->m, n = o->n, k = o->k;
    long lda = o->ta == NoTrans ? m : k, ldb = o->tb == NoTrans ? k : n;
    double *c0 = malloc(m * n * sizeof(double));
#define AT(p, i) (esize == 8 ? ((double *)(p))[i] : (double)((float *)(p))[i])
    for (long i = 0; i < m * n; i++) c0[i] = AT(o->C, i);
    op_call(o);
    double maxerr = 0, maxref = 1e-300;
    for (long j = 0; j < n; j++)
        for (long i = 0; i < m; i++) {
            double acc = 0;
            for (long l = 0; l < k; l++) {
                double a = o->ta == NoTrans ? AT(o->A, i + l * lda) : AT(o->A, l + i * lda);
                double b = o->tb == NoTrans ? AT(o->B, l + j * ldb) : AT(o->B, j + l * ldb);
                acc += a * b;
            }
            double ref = acc + o->beta * c0[i + j * m];
            double err = ref - AT(o->C, i + j * m);
            if (err < 0) err = -err;
            if (err > maxerr) maxerr = err;
            if ((ref < 0 ? -ref : ref) > maxref) maxref = ref < 0 ? -ref : ref;
        }
#undef AT
    free(c0);
    return maxerr / maxref;
}

// Concurrent callers: every thread runs its own op in a loop for the same
// wall-clock window; per-call times of all threads are pooled.
struct caller {
    const struct op *o;
    pthread_barrier_t *bar;
    double min_time;
    int max_reps;
    double *times;
    int reps;
};

static void *caller_main(void *varg) {
    struct caller *c = varg;
    double t0 = now();
    while (now() - t0 < 0.05) op_call(c->o);
    pthread_barrier_wait(c->bar);
    double start = now();
    c->reps = 0;
    while (c->reps < c->max_reps && (c->reps < 3 || now() - start < c->min_time)) {
        double a = now();
        op_call(c->o);
        c->times[c->reps++] = now() - a;
    }
    return NULL;
}

static void run_callers(struct op *ops, int P, double min_time, int max_reps,
                        double *best, double *med, int *reps) {
    if (max_reps > 20000) max_reps = 20000;
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, P);
    pthread_t *th = malloc(P * sizeof *th);
    struct caller *cs = malloc(P * sizeof *cs);
    for (int p = 0; p < P; p++) {
        cs[p] = (struct caller){&ops[p], &bar, min_time, max_reps,
                                malloc(max_reps * sizeof(double)), 0};
        pthread_create(&th[p], NULL, caller_main, &cs[p]);
    }
    int total = 0;
    for (int p = 0; p < P; p++) { pthread_join(th[p], NULL); total += cs[p].reps; }
    double *all = malloc(total * sizeof(double));
    int j = 0;
    for (int p = 0; p < P; p++) {
        memcpy(all + j, cs[p].times, cs[p].reps * sizeof(double));
        j += cs[p].reps;
        free(cs[p].times);
    }
    qsort(all, total, sizeof(double), cmp_double);
    *best = all[0];
    *med = all[total / 2];
    *reps = total;
    free(all); free(cs); free(th);
    pthread_barrier_destroy(&bar);
}

int main(int argc, char **argv) {
    const char *lib = NULL, *prefix = "", *suffix = "", *label = "", *init = "serial";
    char dtype = 'd';
    double min_time = 0.5;
    int max_reps = 100000, threads = 0, header = 0, callers = 0, check = 0;
    double beta = 0;
    int i = 1;
    for (; i < argc && strncmp(argv[i], "--", 2) == 0; i += 2) {
        if (!strcmp(argv[i], "--header")) { header = 1; i--; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
        const char *v = argv[i + 1];
        if (!strcmp(argv[i], "--lib")) lib = v;
        else if (!strcmp(argv[i], "--prefix")) prefix = v;
        else if (!strcmp(argv[i], "--suffix")) suffix = v;
        else if (!strcmp(argv[i], "--dtype")) dtype = v[0];
        else if (!strcmp(argv[i], "--min-time")) min_time = atof(v);
        else if (!strcmp(argv[i], "--max-reps")) max_reps = atoi(v);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(v);
        else if (!strcmp(argv[i], "--label")) label = v;
        else if (!strcmp(argv[i], "--init")) init = v;
        else if (!strcmp(argv[i], "--callers")) callers = atoi(v);
        else if (!strcmp(argv[i], "--beta")) beta = atof(v);
        else if (!strcmp(argv[i], "--check")) check = atoi(v);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!lib) { fprintf(stderr, "--lib is required\n"); return 2; }
    // Env fallback so sweep.py variants can select e.g. Accelerate's $NEWLAPACK.
    if (!*suffix && getenv("GEMMBENCH_SUFFIX")) suffix = getenv("GEMMBENCH_SUFFIX");
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
    // Accelerate (macOS 15+): 0 = multi-threaded, 1 = single-threaded.
    int (*blas_set_threading)(unsigned) = dlsym(h, "BLASSetThreading");
    const char *bt = getenv("GEMMBENCH_BLAS_THREADING");
    if (blas_set_threading && bt) blas_set_threading((unsigned)atoi(bt));
    unsigned (*blas_get_threading)(void) = dlsym(h, "BLASGetThreading");
    if (blas_get_threading) fprintf(stderr, "# BLASGetThreading: %u\n", blas_get_threading());
    int nthr = get_threads ? get_threads() : threads;
    fprintf(stderr, "# library threads: %d\n", nthr);

    char name[64];
#define LOAD(var, base) \
    snprintf(name, sizeof name, "cblas_" base "%s", suffix); \
    var = sym(h, prefix, name);
    LOAD(dgemm, "dgemm") LOAD(sgemm, "sgemm") LOAD(dsyrk, "dsyrk")
    LOAD(ssyrk, "ssyrk") LOAD(dgemv, "dgemv") LOAD(sgemv, "sgemv")
#undef LOAD
    if (!dgemm || !sgemm || !dsyrk || !ssyrk || !dgemv || !sgemv) {
        fprintf(stderr, "cblas symbols not found\n");
        return 1;
    }

    if (header)
        printf("label,dtype,m,n,k,trans,threads,reps,best_s,median_s,"
               "gflops_best,gflops_median\n");
    int init_threads = !strcmp(init, "parallel") ? (nthr > 0 ? nthr : 1) : 1;
    int esize = dtype == 'd' ? 8 : 4;
    for (; i < argc; i++) {
        struct op o = {.routine = 'g', .dtype = dtype, .k = 1, .beta = beta};
        char tt[3] = "NN";
        const char *spec = argv[i];
        int ok;
        if (!strncmp(spec, "syrk:", 5)) {
            o.routine = 's';
            tt[0] = 'N'; tt[1] = 0;
            ok = sscanf(spec + 5, "%ldx%ld:%1s", &o.n, &o.k, tt) >= 2;
            o.m = o.n;
        } else if (!strncmp(spec, "gemv:", 5)) {
            o.routine = 'v';
            tt[0] = 'N'; tt[1] = 0;
            ok = sscanf(spec + 5, "%ldx%ld:%1s", &o.m, &o.n, tt) >= 2;
        } else {
            ok = sscanf(spec, "%ldx%ldx%ld:%2s", &o.m, &o.n, &o.k, tt) >= 3;
        }
        if (!ok) { fprintf(stderr, "bad shape %s\n", spec); return 2; }
        o.ta = tt[0] == 'T' ? Trans : NoTrans;
        o.tb = tt[1] == 'T' ? Trans : NoTrans;
        size_t na, nb, nc;
        if (o.routine == 'g') { na = o.m * o.k; nb = o.k * o.n; nc = o.m * o.n; }
        else if (o.routine == 's') { na = o.n * o.k; nb = 1; nc = o.n * o.n; }
        else { na = o.m * o.n; nb = o.ta == NoTrans ? o.n : o.m; nc = o.ta == NoTrans ? o.m : o.n; }
        if (o.routine == 'g') snprintf(name, sizeof name, "%s", tt);
        else snprintf(name, sizeof name, "%s%s", o.routine == 's' ? "syrk" : "gemv", tt);

        int P = callers > 0 ? callers : 1;
        struct op *ops = malloc(P * sizeof *ops);
        for (int p = 0; p < P; p++) {
            ops[p] = o;
            ops[p].A = alloc_init(na, esize, init_threads, 1 + 3 * p);
            ops[p].B = alloc_init(nb, esize, init_threads, 2 + 3 * p);
            ops[p].C = alloc_init(nc, esize, init_threads, 3 + 3 * p);
        }
        if (check && o.routine == 'g') {
            double rel = check_gemm(&ops[0], esize);
            printf("check,%c,%ld,%ld,%ld,%s,beta=%g,relerr=%.3g,%s\n", dtype, o.m, o.n,
                   o.k, name, beta, rel, rel < (dtype == 'd' ? 1e-13 : 1e-5) ? "OK" : "FAIL");
            fflush(stdout);
            for (int p = 0; p < P; p++) { free(ops[p].A); free(ops[p].B); free(ops[p].C); }
            free(ops);
            continue;
        }
        double flops = op_flops(&o);
        double best, med;
        int reps;
        if (callers > 0) {
            run_callers(ops, P, min_time, max_reps, &best, &med, &reps);
            flops *= P;  // aggregate throughput of the P concurrent callers
        } else {
            // Warm up the thread pool, caches and page tables for >= 50 ms.
            o = ops[0];
            double t0 = now();
            int w = 0;
            do { op_call(&o); w++; } while (now() - t0 < 0.05 && w < max_reps);
            int cap = 1 << 16;
            double *times = malloc(cap * sizeof(double));
            reps = 0;
            double start = now();
            while (reps < max_reps && (reps < 3 || now() - start < min_time)) {
                double a = now();
                op_call(&o);
                double b = now();
                if (reps == cap) { cap *= 2; times = realloc(times, cap * sizeof(double)); }
                times[reps++] = b - a;
            }
            qsort(times, reps, sizeof(double), cmp_double);
            best = times[0];
            med = times[reps / 2];
            free(times);
        }
        printf("%s,%c,%ld,%ld,%ld,%s,%d,%d,%.9g,%.9g,%.4f,%.4f\n", label, dtype,
               o.m, o.n, o.k, name, callers > 0 ? -P : nthr, reps, best, med,
               flops / best * 1e-9, flops / med * 1e-9);
        fflush(stdout);
        for (int p = 0; p < P; p++) { free(ops[p].A); free(ops[p].B); free(ops[p].C); }
        free(ops);
    }
    return 0;
}
