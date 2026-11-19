// Exact check of gemm and syrk under concurrent callers: P threads call
// cblas_{s,d}gemm / cblas_{s,d}syrk at the same time on small-integer inputs,
// where every sum is exact, and compare to a naive reference, including the
// padding between columns and the triangle syrk must not touch.
// usage: conc_check LIB P ITERS [SEED]   (library threads from the environment)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ColMajor = 102, NoTrans = 111, Trans = 112, Upper = 121, Lower = 122 };
enum { PAD = 3 };

typedef void (*dgemm_t)(int, int, int, int, int, int, double, const double *, int,
                        const double *, int, double, double *, int);
typedef void (*sgemm_t)(int, int, int, int, int, int, float, const float *, int,
                        const float *, int, float, float *, int);
typedef void (*dsyrk_t)(int, int, int, int, int, double, const double *, int, double,
                        double *, int);
typedef void (*ssyrk_t)(int, int, int, int, int, float, const float *, int, float,
                        float *, int);
static dgemm_t dgemm;
static sgemm_t sgemm;
static dsyrk_t dsyrk;
static ssyrk_t ssyrk;

// {m, n, k}; m = 0 means syrk with n x n C.
static const int shapes[][3] = {
    {37, 45, 2000}, {64, 64, 20000}, {128, 128, 5000}, {10, 200, 30000},
    {200, 200, 8000}, {300, 300, 600}, {33, 31, 700},
    {0, 32, 20000}, {0, 64, 5000}, {0, 100, 20000}, {0, 200, 6000},
    {0, 300, 2000}, {0, 45, 1500}, {0, 130, 40000},
};
enum { NSHAPES = sizeof shapes / sizeof shapes[0] };

static unsigned rnd(unsigned *s) { return *s = *s * 1664525u + 1013904223u; }

static void put(void *buf, int dbl, long i, double v) {
    if (dbl) ((double *)buf)[i] = v; else ((float *)buf)[i] = (float)v;
}
static double get(const void *buf, int dbl, long i) {
    return dbl ? ((const double *)buf)[i] : (double)((const float *)buf)[i];
}

// Logical op(X) (rows x k, row-contiguous int8) stored column-major as
// rows x k (trans = 0) or k x rows (trans = 1) with leading dimension ld.
static void *layout(const int8_t *h, int rows, int k, int trans, int dbl, int *ld) {
    int r = trans ? k : rows, c = trans ? rows : k;
    *ld = r + PAD;
    void *buf = malloc((size_t)*ld * c * (dbl ? 8 : 4));
    for (long i = 0; i < (long)*ld * c; i++) put(buf, dbl, i, 7);
    for (int i = 0; i < rows; i++)
        for (int l = 0; l < k; l++)
            put(buf, dbl, trans ? l + (long)i * *ld : i + (long)l * *ld, h[(long)i * k + l]);
    return buf;
}

static long dot(const int8_t *x, const int8_t *y, int k) {
    long s = 0;
    for (int l = 0; l < k; l++) s += x[l] * y[l];
    return s;
}

struct caller { int id, iters; unsigned seed; long calls, bad; };

static int one_call(unsigned *seed) {
    const int *sh = shapes[(rnd(seed) >> 8) % NSHAPES];
    int syrk = sh[0] == 0, m = syrk ? sh[1] : sh[0], n = sh[1], k = sh[2];
    int dbl = (rnd(seed) >> 8) & 1, ta = (rnd(seed) >> 8) & 1, tb = (rnd(seed) >> 8) & 1;
    int lower = (rnd(seed) >> 8) & 1;
    double alpha = -2, beta = ((rnd(seed) >> 8) & 1) ? 0.5 : 0;
    int8_t *ah = malloc((size_t)m * k), *bh = syrk ? ah : malloc((size_t)n * k);
    for (long i = 0; i < (long)m * k; i++) ah[i] = (int8_t)((rnd(seed) >> 8) % 5) - 2;
    if (!syrk)
        for (long i = 0; i < (long)n * k; i++) bh[i] = (int8_t)((rnd(seed) >> 8) % 5) - 2;
    int lda, ldb = 0, ldc = m + PAD;
    void *a = layout(ah, m, k, ta, dbl, &lda);
    // op(B) is k x n: logical B^T rows are bh, so B is stored with the other flag.
    void *b = syrk ? NULL : layout(bh, n, k, !tb, dbl, &ldb);
    size_t csize = (size_t)ldc * n * (dbl ? 8 : 4);
    void *c = malloc(csize), *ref = malloc(csize);
    for (long i = 0; i < (long)ldc * n; i++) put(c, dbl, i, i % 11);
    if (beta == 0)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) put(c, dbl, i + (long)j * ldc, NAN);
    memcpy(ref, c, csize);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            if (syrk && (lower ? i < j : i > j)) continue;
            long p = i + (long)j * ldc;
            double old = beta == 0 ? 0 : beta * get(ref, dbl, p);
            put(ref, dbl, p, alpha * (double)dot(ah + (long)i * k, bh + (long)j * k, k) + old);
        }
    int T[2] = {NoTrans, Trans};
    if (syrk) {
        if (dbl) dsyrk(ColMajor, lower ? Lower : Upper, T[ta], n, k, alpha, a, lda, beta, c, ldc);
        else ssyrk(ColMajor, lower ? Lower : Upper, T[ta], n, k, (float)alpha, a, lda, (float)beta, c, ldc);
    } else {
        if (dbl) dgemm(ColMajor, T[ta], T[tb], m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        else sgemm(ColMajor, T[ta], T[tb], m, n, k, (float)alpha, a, lda, b, ldb, (float)beta, c, ldc);
    }
    long bad = 0;
    for (long i = 0; i < (long)ldc * n; i++) {
        double x = get(c, dbl, i), y = get(ref, dbl, i);
        if (!(x == y || (isnan(x) && isnan(y)))) bad++;
    }
    if (bad)
        fprintf(stderr, "MISMATCH %s %c m=%d n=%d k=%d ta=%d tb=%d lower=%d beta=%g: %ld entries\n",
                syrk ? "syrk" : "gemm", dbl ? 'd' : 's', m, n, k, ta, tb, lower, beta, bad);
    free(a); free(b); free(c); free(ref); free(ah);
    if (!syrk) free(bh);
    return bad != 0;
}

static void *caller_main(void *varg) {
    struct caller *c = varg;
    for (int i = 0; i < c->iters; i++) { c->bad += one_call(&c->seed); c->calls++; }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: conc_check LIB P ITERS [SEED]\n"); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    dgemm = dlsym(h, "cblas_dgemm"); sgemm = dlsym(h, "cblas_sgemm");
    dsyrk = dlsym(h, "cblas_dsyrk"); ssyrk = dlsym(h, "cblas_ssyrk");
    int P = atoi(argv[2]), iters = atoi(argv[3]);
    unsigned seed = argc > 4 ? (unsigned)atoi(argv[4]) : 1;
    pthread_t *th = malloc(P * sizeof *th);
    struct caller *cs = calloc(P, sizeof *cs);
    for (int p = 0; p < P; p++) {
        cs[p] = (struct caller){p, iters, seed * 7919u + p * 104729u + 1, 0, 0};
        pthread_create(&th[p], NULL, caller_main, &cs[p]);
    }
    long calls = 0, bad = 0;
    for (int p = 0; p < P; p++) { pthread_join(th[p], NULL); calls += cs[p].calls; bad += cs[p].bad; }
    printf("conc_check callers=%d calls=%ld failures=%ld\n", P, calls, bad);
    return bad != 0;
}
