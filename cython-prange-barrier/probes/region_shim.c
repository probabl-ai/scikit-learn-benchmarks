// LD_PRELOAD library: where does the time of each OpenMP region go, on an unmodified build?
// Build: gcc -O2 -fPIC -shared region_shim.c -o region_shim.so -ldl
// Run: LD_PRELOAD=region_shim.so[:libomp.so] SHIM_OUT=file python ...
// All values are totals in seconds, by region. For a thread, `work` starts after its last GOMP_barrier
// of the region if there is one, so builds with and without the start barrier compare.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_THREADS 512
#define MAX_CPUS 1024
#define MAX_SITES 256

typedef void (*fn_t)(void *);
typedef void (*par_t)(fn_t, void *, unsigned, unsigned);
typedef void (*bar_t)(void);

struct per_thread {
    double entry, after, end;
    long region;
    int cpu_after, cpu_end;
    char pad[64];
} __attribute__((aligned(64)));

struct site {
    fn_t fn;
    long n;
    double wall, wake, start_spread, work_mean, work_max, exit_wait;
    double stacked, moved, team;
};

static par_t real_par;
static bar_t real_bar;
static struct per_thread per[MAX_THREADS];
static struct site sites[MAX_SITES];
static int n_sites;
static fn_t cur_fn;
static void *cur_data;
static long region_id;
static double gap, last_end;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void wrapper(void *unused) {
    (void)unused;
    struct per_thread *p = &per[omp_get_thread_num()];
    p->region = region_id;
    p->entry = p->after = now();
    p->cpu_after = sched_getcpu();
    cur_fn(cur_data);
    p->end = now();
    p->cpu_end = sched_getcpu();
}

void GOMP_barrier(void) {
    if (!real_bar) real_bar = (bar_t)dlsym(RTLD_NEXT, "GOMP_barrier");
    real_bar();
    if (omp_in_parallel()) {
        struct per_thread *p = &per[omp_get_thread_num()];
        p->after = now();
        p->cpu_after = sched_getcpu();
    }
}

static struct site *get_site(fn_t fn) {
    for (int i = 0; i < n_sites; i++)
        if (sites[i].fn == fn) return &sites[i];
    if (n_sites == MAX_SITES) return NULL;
    sites[n_sites].fn = fn;
    return &sites[n_sites++];
}

void GOMP_parallel(fn_t fn, void *data, unsigned n_threads, unsigned flags) {
    if (!real_par) real_par = (par_t)dlsym(RTLD_NEXT, "GOMP_parallel");
    if (omp_in_parallel()) {
        real_par(fn, data, n_threads, flags);
        return;
    }
    double t0 = now();
    if (last_end > 0) gap += t0 - last_end;
    cur_fn = fn;
    cur_data = data;
    region_id++;
    real_par(wrapper, NULL, n_threads, flags);
    double t1 = now();
    last_end = t1;

    struct site *s = get_site(fn);
    if (!s) return;
    int team = 0, stacked = 0, moved = 0;
    double wake = 0, a_min = 1e300, a_max = 0, work_sum = 0, work_max = 0, end_max = 0;
    static int count[MAX_CPUS];
    memset(count, 0, sizeof(count));
    for (int i = 0; i < MAX_THREADS; i++) {
        struct per_thread *p = &per[i];
        if (p->region != region_id) continue;
        team++;
        if (p->entry - t0 > wake) wake = p->entry - t0;
        if (p->after < a_min) a_min = p->after;
        if (p->after > a_max) a_max = p->after;
        work_sum += p->end - p->after;
        if (p->end - p->after > work_max) work_max = p->end - p->after;
        if (p->end > end_max) end_max = p->end;
        moved += p->cpu_after != p->cpu_end;
        if (p->cpu_after >= 0 && p->cpu_after < MAX_CPUS) count[p->cpu_after]++;
    }
    for (int c = 0; c < MAX_CPUS; c++)
        if (count[c] > 1) stacked += count[c];
    s->n++;
    s->wall += t1 - t0;
    s->wake += wake;
    s->start_spread += a_max - a_min;
    s->work_mean += work_sum / (team ? team : 1);
    s->work_max += work_max;
    s->exit_wait += t1 - end_max;
    s->stacked += stacked;
    s->moved += moved;
    s->team += team;
}

static int by_wall(const void *a, const void *b) {
    double d = ((const struct site *)b)->wall - ((const struct site *)a)->wall;
    return (d > 0) - (d < 0);
}

__attribute__((destructor)) static void report(void) {
    const char *path = getenv("SHIM_OUT");
    if (!path || !n_sites) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    struct site t;
    memset(&t, 0, sizeof(t));
    qsort(sites, n_sites, sizeof(struct site), by_wall);
    for (int i = 0; i < n_sites; i++) {
        struct site *s = &sites[i];
        t.n += s->n; t.wall += s->wall; t.wake += s->wake; t.start_spread += s->start_spread;
        t.work_mean += s->work_mean; t.work_max += s->work_max; t.exit_wait += s->exit_wait;
        t.stacked += s->stacked; t.moved += s->moved; t.team += s->team;
    }
    double n = t.n ? t.n : 1;
    fprintf(f, "site=TOTAL n=%ld wall=%.4f gap=%.4f wake=%.4f spread=%.4f work_mean=%.4f work_max=%.4f exit=%.4f team=%.2f stacked=%.3f moved=%.3f\n",
            t.n, t.wall, gap, t.wake, t.start_spread, t.work_mean, t.work_max, t.exit_wait, t.team / n, t.stacked / n, t.moved / n);
    for (int i = 0; i < n_sites && i < 12; i++) {
        struct site *s = &sites[i];
        Dl_info info;
        const char *lib = "?";
        long off = 0;
        if (dladdr((void *)s->fn, &info) && info.dli_fname) {
            lib = strrchr(info.dli_fname, '/') ? strrchr(info.dli_fname, '/') + 1 : info.dli_fname;
            off = (char *)s->fn - (char *)info.dli_fbase;
        }
        char name[64];
        snprintf(name, sizeof(name), "%.*s", (int)strcspn(lib, "."), lib);
        double k = s->n ? s->n : 1;
        fprintf(f, "site=%s+%lx n=%ld wall=%.4f gap=0 wake=%.4f spread=%.4f work_mean=%.4f work_max=%.4f exit=%.4f team=%.2f stacked=%.3f moved=%.3f\n",
                name, off, s->n, s->wall, s->wake, s->start_spread, s->work_mean, s->work_max, s->exit_wait, s->team / k, s->stacked / k, s->moved / k);
    }
    fclose(f);
}
