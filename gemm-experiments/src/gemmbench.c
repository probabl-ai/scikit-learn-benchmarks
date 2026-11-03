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
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#include <unistd.h>

#ifdef __APPLE__
// macOS has no pthread barriers.
typedef struct { pthread_mutex_t mu; pthread_cond_t cv; int n, count, gen; } pthread_barrier_t;
static int pthread_barrier_init(pthread_barrier_t *b, void *attr, int n) {
    (void)attr;
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->n = n; b->count = 0; b->gen = 0;
    return 0;
}
static int pthread_barrier_wait(pthread_barrier_t *b) {
    pthread_mutex_lock(&b->mu);
    int gen = b->gen;
    if (++b->count == b->n) { b->count = 0; b->gen++; pthread_cond_broadcast(&b->cv); }
    else while (gen == b->gen) pthread_cond_wait(&b->cv, &b->mu);
    pthread_mutex_unlock(&b->mu);
    return 0;
}
static int pthread_barrier_destroy(pthread_barrier_t *b) {
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
    return 0;
}
#endif

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

struct fop;

struct op {
    char routine;  // 'g' gemm, 's' syrk, 'v' gemv, 'f' Fortran BLAS/LAPACK (struct fop)
    char dtype;
    long m, n, k;
    int ta, tb;
    double beta;
    void *A, *B, *C;
    struct fop *f;
};

// Float64 level-1/2/3 and LAPACK routines called through their Fortran
// symbols (dpotrf_, ...), which OpenBLAS and MKL both export. Spec NAME:DIMS,
// e.g. dot:100000, ger:100000x20, potrf:2000, gelsd:100000x100x1. LAPACK
// routines overwrite their inputs, so fop_prep restores them before each call,
// outside the timed region. The flop counts of the LAPACK routines are the
// usual nominal ones (syevd/syevr: the tridiagonal reduction, gesdd/gelsd:
// the bidiagonalization), only meant to compare runs of the same routine.
typedef int fint;
static void *blas_handle;
static const char *blas_prefix = "";

enum { F_DOT, F_AXPY, F_NRM2, F_ASUM, F_SCAL, F_GER, F_SYMV, F_TRSV, F_TRSM, F_POTRF,
       F_POSV, F_GETRF, F_GESV, F_GEQRF, F_GEQRT, F_GEQRT3, F_SYEVD, F_SYEVR, F_GESDD, F_GELSD,
       F_NKIND };
static const char *fnames[F_NKIND] = {
    "dot", "axpy", "nrm2", "asum", "scal", "ger", "symv", "trsv", "trsm", "potrf",
    "posv", "getrf", "gesv", "geqrf", "geqrt", "geqrt3", "syevd", "syevr", "gesdd", "gelsd"};

struct fop {
    int kind;
    void *fn;
    long a_n, b_n;     // elements of A and B restored before each call (0: none)
    long na, nb, nc;   // sizes of A, B, C
    double *A0, *B0;   // pristine copies
    double *W, *Z, *U, *VT, *work;
    fint *iwork, *ipiv, lwork, liwork;
    volatile double sink;
};

static int fop_kind(const char *spec, size_t len) {
    for (int k = 0; k < F_NKIND; k++)
        if (strlen(fnames[k]) == len && !strncmp(spec, fnames[k], len)) return k;
    return -1;
}

static double fop_flops(const struct op *o) {
    double m = o->m, n = o->n, r = o->k;
    switch (o->f->kind) {
    case F_DOT: case F_AXPY: case F_NRM2: return 2 * n;
    case F_ASUM: case F_SCAL: return n;
    case F_GER: return 2 * m * n;
    case F_SYMV: return 2 * n * n;
    case F_TRSV: return n * n;
    case F_TRSM: return m * m * n;
    case F_POTRF: return n * n * n / 3;
    case F_POSV: return n * n * n / 3 + 2 * n * n * r;
    case F_GETRF: return m * n * n - n * n * n / 3;
    case F_GESV: return 2 * n * n * n / 3 + 2 * n * n * r;
    case F_GEQRF: case F_GEQRT: case F_GEQRT3: return 2 * m * n * n - 2 * n * n * n / 3;
    case F_SYEVD: case F_SYEVR: return 4 * n * n * n / 3;
    default: return 4 * m * n * n - 4 * n * n * n / 3;  // gesdd, gelsd
    }
}

static void fop_prep(const struct op *o) {
    struct fop *f = o->f;
    if (f->a_n) memcpy(o->A, f->A0, f->a_n * sizeof(double));
    if (f->b_n) memcpy(o->B, f->B0, f->b_n * sizeof(double));
}

static void fop_call(const struct op *o) {
    struct fop *f = o->f;
    fint m = o->m, n = o->n, r = o->k, one = 1, info = 0, lda = o->m, ldb;
    double *A = o->A, *B = o->B, *C = o->C, alpha = 1e-3, done = 1.0, dzero = 0.0;
    size_t l = 1;
    switch (f->kind) {
    case F_DOT: f->sink = ((double (*)(fint *, double *, fint *, double *, fint *))f->fn)(&n, A, &one, B, &one); break;
    case F_NRM2: case F_ASUM: f->sink = ((double (*)(fint *, double *, fint *))f->fn)(&n, A, &one); break;
    case F_AXPY: ((void (*)(fint *, double *, double *, fint *, double *, fint *))f->fn)(&n, &alpha, A, &one, B, &one); break;
    case F_SCAL: alpha = -1.0; ((void (*)(fint *, double *, double *, fint *))f->fn)(&n, &alpha, A, &one); break;
    case F_GER:
        ((void (*)(fint *, fint *, double *, double *, fint *, double *, fint *, double *, fint *))f->fn)(
            &m, &n, &alpha, B, &one, C, &one, A, &lda);
        break;
    case F_SYMV:
        ((void (*)(char *, fint *, double *, double *, fint *, double *, fint *, double *, double *, fint *, size_t))f->fn)(
            "U", &n, &done, A, &n, B, &one, &dzero, C, &one, l);
        break;
    case F_TRSV:
        ((void (*)(char *, char *, char *, fint *, double *, fint *, double *, fint *, size_t, size_t, size_t))f->fn)(
            "U", "N", "N", &n, A, &n, B, &one, l, l, l);
        break;
    case F_TRSM:
        ((void (*)(char *, char *, char *, char *, fint *, fint *, double *, double *, fint *, double *, fint *, size_t, size_t, size_t, size_t))f->fn)(
            "L", "U", "N", "N", &m, &n, &done, A, &m, B, &m, l, l, l, l);
        break;
    case F_POTRF:
        ((void (*)(char *, fint *, double *, fint *, fint *, size_t))f->fn)("U", &n, A, &n, &info, l);
        break;
    case F_POSV:
        ((void (*)(char *, fint *, fint *, double *, fint *, double *, fint *, fint *, size_t))f->fn)(
            "U", &n, &r, A, &n, B, &n, &info, l);
        break;
    case F_GETRF:
        ((void (*)(fint *, fint *, double *, fint *, fint *, fint *))f->fn)(&m, &n, A, &m, f->ipiv, &info);
        break;
    case F_GESV:
        ((void (*)(fint *, fint *, double *, fint *, fint *, double *, fint *, fint *))f->fn)(
            &n, &r, A, &n, f->ipiv, B, &n, &info);
        break;
    case F_GEQRF:
        ((void (*)(fint *, fint *, double *, fint *, double *, double *, fint *, fint *))f->fn)(
            &m, &n, A, &m, f->W, f->work, &f->lwork, &info);
        break;
    case F_GEQRT: {
        // dgeqrt(m, n, nb, A, lda, T, ldt, work, info); nb from the third dimension.
        fint nb = r > 1 ? r : 32;
        ((void (*)(fint *, fint *, fint *, double *, fint *, double *, fint *, double *, fint *))f->fn)(
            &m, &n, &nb, A, &m, f->Z, &nb, f->work, &info);
        break;
    }
    case F_GEQRT3:
        ((void (*)(fint *, fint *, double *, fint *, double *, fint *, fint *))f->fn)(&m, &n, A, &m, f->Z, &n, &info);
        break;
    case F_SYEVD:
        ((void (*)(char *, char *, fint *, double *, fint *, double *, double *, fint *, fint *, fint *, fint *, size_t, size_t))f->fn)(
            "V", "U", &n, A, &n, f->W, f->work, &f->lwork, f->iwork, &f->liwork, &info, l, l);
        break;
    case F_SYEVR: {
        fint il = 0, iu = 0, found = 0;
        double vl = 0, vu = 0, abstol = 0;
        ((void (*)(char *, char *, char *, fint *, double *, fint *, double *, double *, fint *, fint *, double *, fint *, double *, double *, fint *, fint *, double *, fint *, fint *, fint *, fint *, size_t, size_t, size_t))f->fn)(
            "V", "A", "U", &n, A, &n, &vl, &vu, &il, &iu, &abstol, &found, f->W, f->Z, &n,
            f->ipiv, f->work, &f->lwork, f->iwork, &f->liwork, &info, l, l, l);
        break;
    }
    case F_GESDD: {
        fint mn = m < n ? m : n;
        ((void (*)(char *, fint *, fint *, double *, fint *, double *, double *, fint *, double *, fint *, double *, fint *, fint *, fint *, size_t))f->fn)(
            "S", &m, &n, A, &m, f->W, f->U, &m, f->VT, &mn, f->work, &f->lwork, f->iwork, &info, l);
        break;
    }
    case F_GELSD: {
        fint rank = 0;
        double rcond = -1;
        ldb = m > n ? m : n;
        ((void (*)(fint *, fint *, fint *, double *, fint *, double *, fint *, double *, double *, fint *, double *, fint *, fint *, fint *))f->fn)(
            &m, &n, &r, A, &m, B, &ldb, f->W, &rcond, &rank, f->work, &f->lwork, f->iwork, &info);
        break;
    }
    }
    if (info != 0 && f->lwork >= 0) { fprintf(stderr, "%s: info=%d\n", fnames[f->kind], (int)info); exit(1); }
}

static void *zalloc(size_t nelem, size_t esize) {
    size_t nbytes = (nelem * esize + 4095) / 4096 * 4096;
    void *p = aligned_alloc(4096, nbytes ? nbytes : 4096);
    if (!p) { perror("aligned_alloc"); exit(1); }
    memset(p, 0, nbytes ? nbytes : 4096);
    return p;
}

static void *sym(void *h, const char *prefix, const char *name);

// Sizes A, B, C for o (m, n, k already parsed), fills them and the workspaces.
static void fop_setup(struct op *o, void *(*init)(size_t, int, int, unsigned), int seed) {
    struct fop *f = o->f;
    long m = o->m, n = o->n, r = o->k;
    char name[32];
    snprintf(name, sizeof name, "d%s_", fnames[f->kind]);
    if (!(f->fn = sym(blas_handle, blas_prefix, name))) { fprintf(stderr, "no symbol %s\n", name); exit(1); }
    long na = 1, nb = 1, nc = 1;
    switch (f->kind) {
    case F_DOT: case F_AXPY: na = nb = n; break;
    case F_NRM2: case F_ASUM: case F_SCAL: na = n; break;
    case F_GER: na = m * n; nb = m; nc = n; break;
    case F_SYMV: case F_TRSV: na = n * n; nb = nc = n; m = n; break;
    case F_TRSM: na = m * m; nb = m * n; break;
    case F_POTRF: case F_SYEVD: case F_SYEVR: na = n * n; m = n; break;
    case F_POSV: case F_GESV: na = n * n; nb = n * r; m = n; break;
    case F_GETRF: case F_GEQRF: case F_GEQRT: case F_GEQRT3: case F_GESDD: na = m * n; break;
    case F_GELSD: na = m * n; nb = (m > n ? m : n) * r; break;
    }
    o->m = m;
    f->na = na; f->nb = nb; f->nc = nc;
    o->A = init(na, 8, 1, seed);
    o->B = init(nb, 8, 1, seed + 1);
    o->C = init(nc, 8, 1, seed + 2);
    double *A = o->A;
    if (f->kind == F_POTRF || f->kind == F_POSV || f->kind == F_SYMV || f->kind == F_SYEVD ||
        f->kind == F_SYEVR)
        for (long j = 0; j < n; j++)
            for (long i = 0; i < j; i++) A[j + i * n] = A[i + j * n];
    // Diagonally dominant, so the factorizations and solves stay well conditioned.
    if (f->kind == F_POTRF || f->kind == F_POSV || f->kind == F_TRSV || f->kind == F_TRSM ||
        f->kind == F_GESV)
        for (long i = 0; i < m; i++) A[i + i * m] += m;
    int restore_a = f->kind >= F_POTRF, restore_b = f->kind == F_TRSV || f->kind == F_TRSM ||
                                                     f->kind == F_POSV || f->kind == F_GESV ||
                                                     f->kind == F_GELSD;
    if (restore_a) { f->a_n = na; f->A0 = zalloc(na, 8); memcpy(f->A0, o->A, na * 8); }
    if (restore_b) { f->b_n = nb; f->B0 = zalloc(nb, 8); memcpy(f->B0, o->B, nb * 8); }
    long mn = m < n ? m : n;
    f->ipiv = zalloc(2 * (m > n ? m : n) + 1, sizeof(fint));
    f->W = zalloc(m > n ? m : n, 8);
    if (f->kind == F_SYEVR || f->kind == F_GEQRT3) f->Z = zalloc(n * n, 8);
    if (f->kind == F_GEQRT) { long nb = r > 1 ? r : 32; f->Z = zalloc(nb * n, 8); f->work = zalloc(nb * n, 8); }
    if (f->kind == F_GESDD) { f->U = zalloc(m * mn, 8); f->VT = zalloc(mn * n, 8); }
    if (f->kind == F_GEQRF || f->kind >= F_SYEVD) {
        // Workspace query.
        double w = 0;
        fint iw = 0;
        f->work = &w; f->iwork = &iw; f->lwork = -1; f->liwork = -1;
        fop_call(o);
        f->lwork = (fint)w + 1;
        f->liwork = iw > 0 ? iw : 1;
        if (f->kind == F_GESDD) f->liwork = 8 * mn;
        if (f->kind == F_GELSD && f->liwork < 1) f->liwork = 1;
        f->work = zalloc(f->lwork, 8);
        f->iwork = zalloc(f->liwork > 8 * mn ? f->liwork : 8 * mn, sizeof(fint));
        fop_prep(o);
    }
}

// Runs o with the current threads and with one thread from the same inputs,
// and returns the largest difference over A, B, C and the scalar result,
// relative to the largest value of each.
static double fop_check(struct op *o, void (*set_threads)(int), int nthr) {
    struct fop *f = o->f;
    double *arr[3] = {o->A, o->B, o->C};
    long len[3] = {f->na, f->nb, f->nc};
    double *in[3], *out[3];
    for (int j = 0; j < 3; j++) {
        in[j] = malloc(len[j] * 8); out[j] = malloc(len[j] * 8);
        memcpy(in[j], arr[j], len[j] * 8);
    }
    fop_prep(o);
    for (int j = 0; j < 3; j++) memcpy(in[j], arr[j], len[j] * 8);
    fop_call(o);
    double sink = f->sink;
    for (int j = 0; j < 3; j++) { memcpy(out[j], arr[j], len[j] * 8); memcpy(arr[j], in[j], len[j] * 8); }
    set_threads(1);
    fop_call(o);
    set_threads(nthr);
    double worst = 0;
    for (int j = 0; j < 3; j++) {
        double maxerr = 0, maxref = 1e-300;
        for (long i = 0; i < len[j]; i++) {
            double e = out[j][i] - arr[j][i], r = arr[j][i];
            if (e != e) e = 1e300;  // NaN
            if (e < 0) e = -e;
            if (r < 0) r = -r;
            if (e > maxerr) maxerr = e;
            if (r > maxref) maxref = r;
        }
        if (maxerr / maxref > worst) worst = maxerr / maxref;
        free(in[j]); free(out[j]);
    }
    double se = sink - f->sink, sr = f->sink;
    se = se < 0 ? -se : se; sr = sr < 0 ? -sr : sr;
    if (sr > 0 && se / sr > worst) worst = se / sr;
    return worst;
}

static double op_flops(const struct op *o) {
    if (o->routine == 'f') return fop_flops(o);
    if (o->routine == 'g') return 2.0 * o->m * o->n * o->k;
    if (o->routine == 's') return 1.0 * o->n * (o->n + 1) * o->k;
    return 2.0 * o->m * o->n;
}

static void op_call(const struct op *o) {
    if (o->routine == 'f') { fop_call(o); return; }
    int d = o->dtype == 'd';
    if (o->routine == 'g') {
        long lda = o->ta == NoTrans ? o->m : o->k;
        long ldb = o->tb == NoTrans ? o->k : o->n;
        if (d) dgemm(ColMajor, o->ta, o->tb, o->m, o->n, o->k, 1.0, o->A, lda, o->B, ldb, o->beta, o->C, o->m);
        else sgemm(ColMajor, o->ta, o->tb, o->m, o->n, o->k, 1.0f, o->A, lda, o->B, ldb, (float)o->beta, o->C, o->m);
    } else if (o->routine == 's') {
        long lda = o->ta == NoTrans ? o->n : o->k;
        if (d) dsyrk(ColMajor, Upper, o->ta, o->n, o->k, 1.0, o->A, lda, o->beta, o->C, o->n);
        else ssyrk(ColMajor, Upper, o->ta, o->n, o->k, 1.0f, o->A, lda, (float)o->beta, o->C, o->n);
    } else {
        if (d) dgemv(ColMajor, o->ta, o->m, o->n, 1.0, o->A, o->m, o->B, 1, o->beta, o->C, 1);
        else sgemv(ColMajor, o->ta, o->m, o->n, 1.0f, o->A, o->m, o->B, 1, (float)o->beta, o->C, 1);
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
static double check_gemv(struct op *o, int esize);
static double check_syrk(struct op *o, int esize);

static double check_gemm(struct op *o, int esize) {
    if (o->routine == 'v') return check_gemv(o, esize);
    if (o->routine == 's') return check_syrk(o, esize);
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

// Same for gemv: y = A x (N) or A^T x (T), A m x n, on a copy of y.
static double check_gemv(struct op *o, int esize) {
    long m = o->m, n = o->n, ny = o->ta == NoTrans ? m : n;
    double *y0 = malloc(ny * sizeof(double));
#define AT(p, i) (esize == 8 ? ((double *)(p))[i] : (double)((float *)(p))[i])
    for (long i = 0; i < ny; i++) y0[i] = AT(o->C, i);
    op_call(o);
    double maxerr = 0, maxref = 1e-300;
    for (long i = 0; i < ny; i++) {
        double acc = 0;
        if (o->ta == NoTrans) for (long j = 0; j < n; j++) acc += AT(o->A, i + j * m) * AT(o->B, j);
        else for (long j = 0; j < m; j++) acc += AT(o->A, j + i * m) * AT(o->B, j);
        double ref = acc + o->beta * y0[i];
        double err = ref - AT(o->C, i);
        if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        if ((ref < 0 ? -ref : ref) > maxref) maxref = ref < 0 ? -ref : ref;
    }
#undef AT
    free(y0);
    return maxerr / maxref;
}

// Same for syrk: upper triangle of C = op(A) op(A)^T + beta C.
static double check_syrk(struct op *o, int esize) {
    long n = o->n, k = o->k, lda = o->ta == NoTrans ? n : k;
#define AT(p, i) (esize == 8 ? ((double *)(p))[i] : (double)((float *)(p))[i])
    double *c0 = malloc(n * n * sizeof(double));
    for (long i = 0; i < n * n; i++) c0[i] = AT(o->C, i);
    op_call(o);
    double maxerr = 0, maxref = 1e-300;
    for (long j = 0; j < n; j++)
        for (long i = 0; i <= j; i++) {
            double acc = 0;
            for (long l = 0; l < k; l++)
                acc += o->ta == NoTrans ? AT(o->A, i + l * lda) * AT(o->A, j + l * lda)
                                        : AT(o->A, l + i * lda) * AT(o->A, l + j * lda);
            acc += o->beta * c0[i + j * n];
            double err = acc - AT(o->C, i + j * n);
            if (err < 0) err = -err;
            if (err > maxerr) maxerr = err;
            if ((acc < 0 ? -acc : acc) > maxref) maxref = acc < 0 ? -acc : acc;
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
    while (now() - t0 < 0.05) {
        if (c->o->f) fop_prep(c->o);
        op_call(c->o);
    }
    pthread_barrier_wait(c->bar);
    double start = now();
    c->reps = 0;
    while (c->reps < c->max_reps && (c->reps < 3 || now() - start < c->min_time)) {
        if (c->o->f) fop_prep(c->o);
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

#ifdef __linux__
// GEMMBENCH_SAMPLE_PLACEMENT=1: a thread samples every 20 ms on which CPU
// each running thread (state R) sits while a shape is timed, to see threads stacked on one CPU or
// sharing a core while other cores idle.
#define MAXCPU 1024
static int core_of[MAXCPU];
static volatile int sampling = 1, timing = 0;
static double s_samples, s_running, s_cpus, s_cores;

static void *placement_sampler(void *arg) {
    (void)arg;
    for (int c = 0; c < MAXCPU; c++) {
        char p[128];
        int core = -1, pkg = 0;
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        FILE *f = fopen(p, "r");
        if (f) { if (fscanf(f, "%d", &core) != 1) core = -1; fclose(f); }
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        if ((f = fopen(p, "r"))) { if (fscanf(f, "%d", &pkg) != 1) pkg = 0; fclose(f); }
        core_of[c] = core < 0 ? -1 : pkg * 4096 + core;
    }
    char dirpath[64], path[512], line[1024];
    snprintf(dirpath, sizeof dirpath, "/proc/%d/task", (int)getpid());
    pid_t self = (pid_t)syscall(SYS_gettid);
    static unsigned char cpu_seen[MAXCPU];
    static int cores[MAXCPU];
    while (sampling) {
        struct timespec ts = {0, 20000000};
        nanosleep(&ts, NULL);
        if (!timing) continue;
        memset(cpu_seen, 0, sizeof cpu_seen);
        int running = 0, ncpu = 0, ncore = 0;
        DIR *dir = opendir(dirpath);
        struct dirent *e;
        while (dir && (e = readdir(dir))) {
            if (e->d_name[0] == '.' || atoi(e->d_name) == self) continue;
            snprintf(path, sizeof path, "%s/%s/stat", dirpath, e->d_name);
            FILE *f = fopen(path, "r");
            if (!f) continue;
            char *ok = fgets(line, sizeof line, f);
            fclose(f);
            char *q = ok ? strrchr(line, ')') : NULL;
            if (!q || q[2] != 'R') continue;
            // fields after the comm: state is field 3, processor is field 39
            int field = 2, cpu = -1;
            for (char *t = q + 1; *t; t++)
                if (*t == ' ' && ++field == 39) { cpu = atoi(t + 1); break; }
            if (cpu < 0 || cpu >= MAXCPU) continue;
            running++;
            if (!cpu_seen[cpu]++) {
                ncpu++;
                int k, c = core_of[cpu];
                for (k = 0; k < ncore && cores[k] != c; k++) {}
                if (k == ncore) cores[ncore++] = c;
            }
        }
        if (dir) closedir(dir);
        s_samples++; s_running += running; s_cpus += ncpu; s_cores += ncore;
    }
    return NULL;
}
#endif

// Sets "K=V;K=V" in the environment, as a Python caller would with os.environ
// once the BLAS library is loaded.
static void apply_env(const char *spec) {
    char *buf = strdup(spec), *save = NULL;
    for (char *kv = strtok_r(buf, ";", &save); kv; kv = strtok_r(NULL, ";", &save)) {
        char *eq = strchr(kv, '=');
        if (eq) { *eq = 0; setenv(kv, eq + 1, 1); }
    }
    free(buf);
}

static int count_cpus(const char *list) {
    int n = 0, a, b, len;
    while (sscanf(list, "%d%n", &a, &len) == 1) {
        list += len;
        b = a;
        if (*list == '-' && sscanf(list + 1, "%d%n", &b, &len) == 1) list += 1 + len;
        n += b - a + 1;
        if (*list != ',') break;
        list++;
    }
    return n;
}

// How many of our threads are bound to at most one core (2 CPUs), how often
// they slept, were preempted or migrated, and which OpenMP runtime is loaded
// (Linux /proc).
static void report_affinity(void) {
    char dirpath[64], path[512], line[8192], omp[512] = "none";
    int tasks = 0, bound = 0;
    long vcsw = 0, ivcsw = 0, migr = 0, n;
    snprintf(dirpath, sizeof dirpath, "/proc/%d/task", (int)getpid());
    DIR *dir = opendir(dirpath);
    if (!dir) return;
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (e->d_name[0] == '.') continue;
        snprintf(path, sizeof path, "%s/%s/status", dirpath, e->d_name);
        FILE *st = fopen(path, "r");
        if (!st) continue;
        while (fgets(line, sizeof line, st))
            if (!strncmp(line, "Cpus_allowed_list:", 18)) {
                tasks++;
                if (count_cpus(line + 18 + strspn(line + 18, " \t")) <= 2) bound++;
            } else if (sscanf(line, "voluntary_ctxt_switches: %ld", &n) == 1) {
                vcsw += n;
            } else if (sscanf(line, "nonvoluntary_ctxt_switches: %ld", &n) == 1) {
                ivcsw += n;
            }
        fclose(st);
        snprintf(path, sizeof path, "%s/%s/sched", dirpath, e->d_name);
        if ((st = fopen(path, "r"))) {
            while (fgets(line, sizeof line, st))
                if (sscanf(line, "se.nr_migrations : %ld", &n) == 1) migr += n;
            fclose(st);
        }
    }
    closedir(dir);
    FILE *maps = fopen("/proc/self/maps", "r");
    while (maps && fgets(line, sizeof line, maps)) {
        char *p = strchr(line, '/');
        if (p && strstr(p, "omp") && strstr(p, ".so")) {
            p[strcspn(p, "\n")] = 0;
            const char *base = strrchr(p, '/') + 1;
            if (!strstr(omp, base)) {
                if (!strcmp(omp, "none")) omp[0] = 0;
                snprintf(omp + strlen(omp), sizeof omp - strlen(omp), "%s%s",
                         omp[0] ? "," : "", p);
            }
        }
    }
    if (maps) fclose(maps);
    fprintf(stderr, "# affinity: threads=%d bound_to_one_core=%d voluntary_switches=%ld "
            "involuntary_switches=%ld migrations=%ld openmp=%s\n",
            tasks, bound, vcsw, ivcsw, migr, omp);
}

int main(int argc, char **argv) {
    const char *lib = NULL, *prefix = "", *suffix = "", *label = "", *init = "serial";
    char dtype = 'd';
    double min_time = 0.5;
    int max_reps = 100000, threads = 0, header = 0, callers = 0, check = 0, cycle = 0;
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
        else if (!strcmp(argv[i], "--cycle")) cycle = atoi(v);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!lib) { fprintf(stderr, "--lib is required\n"); return 2; }
    // Env fallback so sweep.py variants can select e.g. Accelerate's $NEWLAPACK.
    if (!*suffix && getenv("GEMMBENCH_SUFFIX")) suffix = getenv("GEMMBENCH_SUFFIX");
    void *h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    blas_handle = h;
    blas_prefix = prefix;
    const char *late = getenv("GEMMBENCH_LATE_ENV");
    const char *late_at = getenv("GEMMBENCH_LATE_ENV_AT");
    if (!late_at) late_at = "setter";
    if (late && !strcmp(late_at, "dlopen")) apply_env(late);

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
    // GEMMBENCH_LATE_ENV="K=V;..." is set once the library is loaded, at
    // GEMMBENCH_LATE_ENV_AT: "dlopen" (before any call), "setter" (default,
    // after the thread setter/getter calls above, like threadpoolctl) or
    // "call" (after one threaded dgemm has started the thread pool).
    if (late && strcmp(late_at, "dlopen")) {
        if (!strcmp(late_at, "call")) {
            struct op w = {.routine = 'g', .dtype = 'd', .m = 512, .n = 512, .k = 512,
                           .ta = NoTrans, .tb = NoTrans};
            w.A = alloc_init(512 * 512, 8, 1, 1);
            w.B = alloc_init(512 * 512, 8, 1, 2);
            w.C = alloc_init(512 * 512, 8, 1, 3);
            op_call(&w);
            free(w.A); free(w.B); free(w.C);
        }
        apply_env(late);
    }

#ifdef __linux__
    pthread_t sampler;
    int sample = getenv("GEMMBENCH_SAMPLE_PLACEMENT") != NULL;
    if (sample) pthread_create(&sampler, NULL, placement_sampler, NULL);
#endif

    if (header)
        printf("label,dtype,m,n,k,trans,threads,reps,best_s,median_s,"
               "gflops_best,gflops_median\n");
    struct op *kept = NULL;
    int nkept = 0;
    int init_threads = !strcmp(init, "parallel") ? (nthr > 0 ? nthr : 1) : 1;
    int esize = dtype == 'd' ? 8 : 4;
    for (; i < argc; i++) {
        struct op o = {.routine = 'g', .dtype = dtype, .k = 1, .beta = beta};
        char tt[3] = "NN";
        const char *spec = argv[i];
        int ok;
        const char *colon = strchr(spec, ':');
        int fk = colon ? fop_kind(spec, colon - spec) : -1;
        if (fk >= 0) {
            o.routine = 'f';
            o.f = calloc(1, sizeof *o.f);
            o.f->kind = fk;
            long d[3] = {0, 0, 1};
            int nd = sscanf(colon + 1, "%ldx%ldx%ld", &d[0], &d[1], &d[2]);
            ok = nd >= 1 && dtype == 'd';
            // One dimension: n. Two: m x n, or n x nrhs for posv/gesv.
            if (nd == 1) { o.m = o.n = d[0]; }
            else if (fk == F_POSV || fk == F_GESV) { o.m = o.n = d[0]; o.k = d[1]; }
            else { o.m = d[0]; o.n = d[1]; o.k = d[2]; }
            snprintf(tt, sizeof tt, "%s", "");
        } else if (!strncmp(spec, "syrk:", 5)) {
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
        if (o.routine == 'f') snprintf(name, sizeof name, "%s", fnames[o.f->kind]);
        else if (o.routine == 'g') snprintf(name, sizeof name, "%s", tt);
        else snprintf(name, sizeof name, "%s%s", o.routine == 's' ? "syrk" : "gemv", tt);

        int P = callers > 0 ? callers : 1;
        struct op *ops = malloc(P * sizeof *ops);
        for (int p = 0; p < P; p++) {
            ops[p] = o;
            if (o.routine == 'f') {
                ops[p].f = calloc(1, sizeof *o.f);
                ops[p].f->kind = o.f->kind;
                fop_setup(&ops[p], alloc_init, 1 + 3 * p);
                continue;
            }
            ops[p].A = alloc_init(na, esize, init_threads, 1 + 3 * p);
            ops[p].B = alloc_init(nb, esize, init_threads, 2 + 3 * p);
            ops[p].C = alloc_init(nc, esize, init_threads, 3 + 3 * p);
        }
        if (o.routine == 'f') o.m = ops[0].m;
        if (check && o.routine == 'f') {
            void (*set)(int) = sym(h, prefix, "openblas_set_num_threads");
            if (!set) set = dlsym(h, "MKL_Set_Num_Threads");
            double rel = fop_check(&ops[0], set, nthr);
            printf("check,%c,%ld,%ld,%ld,%s,threads=%d,vs1thread=%.3g,%s\n", dtype, o.m, o.n,
                   o.k, name, nthr, rel, rel < 1e-10 ? "OK" : "FAIL");
            fflush(stdout);
            continue;
        }
        if (check) {
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
#ifdef __linux__
            timing = 1;
#endif
            double t0 = now();
            int w = 0;
            do {
                if (o.f) fop_prep(&o);
                op_call(&o);
                w++;
            } while (now() - t0 < 0.05 && w < max_reps);
            int cap = 1 << 16;
            double *times = malloc(cap * sizeof(double));
            reps = 0;
            double start = now();
            while (reps < max_reps && (reps < 3 || now() - start < min_time)) {
                if (o.f) fop_prep(&o);
                double a = now();
                op_call(&o);
                double b = now();
                if (reps == cap) { cap *= 2; times = realloc(times, cap * sizeof(double)); }
                times[reps++] = b - a;
            }
#ifdef __linux__
            timing = 0;
#endif
            qsort(times, reps, sizeof(double), cmp_double);
            best = times[0];
            med = times[reps / 2];
            free(times);
        }
        printf("%s,%c,%ld,%ld,%ld,%s,%d,%d,%.9g,%.9g,%.4f,%.4f\n", label, dtype,
               o.m, o.n, o.k, name, callers > 0 ? -P : nthr, reps, best, med,
               flops / best * 1e-9, flops / med * 1e-9);
        fflush(stdout);
        if (cycle && callers == 0) {
            kept = realloc(kept, (nkept + 1) * sizeof *kept);
            kept[nkept++] = ops[0];
            free(ops);
            continue;
        }
        for (int p = 0; p < P; p++) { free(ops[p].A); free(ops[p].B); free(ops[p].C); }
        free(ops);
    }
    // --cycle 1: after timing each shape alone, time the shapes called one
    // after the other in a loop (one rep = one call of each), to see the cost
    // of switching between calls that use different numbers of threads.
    if (nkept > 0) {
        double flops = 0;
        for (int j = 0; j < nkept; j++) flops += op_flops(&kept[j]);
        double t0 = now();
        do { for (int j = 0; j < nkept; j++) op_call(&kept[j]); } while (now() - t0 < 0.05);
        int cap = 1 << 16, reps = 0;
        double *times = malloc(cap * sizeof(double));
        double start = now();
        while (reps < max_reps && (reps < 3 || now() - start < min_time)) {
            double a = now();
            for (int j = 0; j < nkept; j++) op_call(&kept[j]);
            double b = now();
            if (reps == cap) { cap *= 2; times = realloc(times, cap * sizeof(double)); }
            times[reps++] = b - a;
        }
        qsort(times, reps, sizeof(double), cmp_double);
        printf("%s,%c,0,0,0,cycle%d,%d,%d,%.9g,%.9g,%.4f,%.4f\n", label, dtype, nkept, nthr,
               reps, times[0], times[reps / 2], flops / times[0] * 1e-9,
               flops / times[reps / 2] * 1e-9);
        fflush(stdout);
        free(times);
        for (int j = 0; j < nkept; j++) { free(kept[j].A); free(kept[j].B); free(kept[j].C); }
    }
    free(kept);
#ifdef __linux__
    if (sample) {
        sampling = 0;
        pthread_join(sampler, NULL);
        if (s_samples > 0)
            fprintf(stderr, "# placement: samples=%.0f running=%.1f cpus=%.1f cores=%.1f\n",
                    s_samples, s_running / s_samples, s_cpus / s_samples, s_cores / s_samples);
    }
#endif
    report_affinity();
    return 0;
}
