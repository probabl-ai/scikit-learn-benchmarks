// Mixed concurrent callers: caller i runs its own dgemm/dsyrk shape in a loop
// for WINDOW seconds (after START_DELAY_i), and reports its calls per second.
// usage: mixbench LIB WINDOW SPEC...   one SPEC per caller:
//   s:N:K[:DELAY]     dsyrk, C n x n, A n x k
//   g:M:N:K[:DELAY]   dgemm NT, C m x n
// DELAY (seconds) staggers the start of a caller. Threads from the environment.
// Output: one CSV row per caller: label,caller,spec,calls,calls_per_s,mean_ms
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ColMajor = 102, NoTrans = 111, Trans = 112, Upper = 121 };
typedef void (*dgemm_t)(int, int, int, int, int, int, double, const double *, int,
                        const double *, int, double, double *, int);
typedef void (*dsyrk_t)(int, int, int, int, int, double, const double *, int, double,
                        double *, int);
static dgemm_t dgemm;
static dsyrk_t dsyrk;
static double window, t_start;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct caller {
    const char *spec;
    int syrk, m, n, k;
    double delay, *a, *b, *c;
    long calls;
    double elapsed;
};

static void call(struct caller *c) {
    if (c->syrk) dsyrk(ColMajor, Upper, NoTrans, c->n, c->k, 1.0, c->a, c->n, 0.0, c->c, c->n);
    else dgemm(ColMajor, NoTrans, Trans, c->m, c->n, c->k, 1.0, c->a, c->m, c->b, c->n, 0.0, c->c, c->m);
}

static void *caller_main(void *varg) {
    struct caller *c = varg;
    while (now() < t_start + c->delay) {
        struct timespec ts = {0, 200000};
        nanosleep(&ts, NULL);
    }
    double t0 = now(), end = t_start + window;
    do { call(c); c->calls++; } while (now() < end);
    c->elapsed = now() - t0;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: mixbench LIB WINDOW SPEC...\n"); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    dgemm = dlsym(h, "cblas_dgemm"); dsyrk = dlsym(h, "cblas_dsyrk");
    window = atof(argv[2]);
    int P = argc - 3;
    struct caller *cs = calloc(P, sizeof *cs);
    pthread_t *th = malloc(P * sizeof *th);
    for (int p = 0; p < P; p++) {
        struct caller *c = &cs[p];
        c->spec = argv[3 + p];
        if (c->spec[0] == 's') {
            c->syrk = 1;
            if (sscanf(c->spec, "s:%d:%d:%lf", &c->n, &c->k, &c->delay) < 2) return 2;
            c->m = c->n;
        } else if (sscanf(c->spec, "g:%d:%d:%d:%lf", &c->m, &c->n, &c->k, &c->delay) < 3) return 2;
        size_t na = (size_t)c->m * c->k, nb = c->syrk ? 0 : (size_t)c->n * c->k;
        c->a = malloc(na * 8); c->b = nb ? malloc(nb * 8) : NULL;
        c->c = malloc((size_t)c->m * c->n * 8);
        for (size_t i = 0; i < na; i++) c->a[i] = (double)(i % 7) - 3;
        for (size_t i = 0; i < nb; i++) c->b[i] = (double)(i % 5) - 2;
        call(c);  // warm-up, one caller at a time
    }
    const char *label = getenv("MIXBENCH_LABEL");
    t_start = now() + 0.05;
    for (int p = 0; p < P; p++) pthread_create(&th[p], NULL, caller_main, &cs[p]);
    for (int p = 0; p < P; p++) pthread_join(th[p], NULL);
    for (int p = 0; p < P; p++)
        printf("%s,%d,%s,%ld,%.3f,%.4f\n", label ? label : "", p, cs[p].spec, cs[p].calls,
               cs[p].calls / cs[p].elapsed, 1e3 * cs[p].elapsed / cs[p].calls);
    return 0;
}
