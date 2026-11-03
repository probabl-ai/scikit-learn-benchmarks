// Like wake.c, but each of the 90 items gathers from its own buffer that
// persists across regions (like a feature's histogram + data, always handled by
// the same thread under schedule(static)). If deep idle flushes the core's
// caches, or the thread wakes up on another core, the gather gets slower.
// Usage: wake2 ITERS ITEM_KB PASSES GAP_US...
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_ITEMS 90
#define MAX_T 64

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}
static volatile double sink;
static void spin(long k) {
    double x = 1.0;
    for (long i = 0; i < k; i++) x = x * 1.0000001 + 1e-9;
    sink = x;
}
static int cmp(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}
static double pct(double *v, int n, double p) {
    qsort(v, n, sizeof(double), cmp);
    return v[(int)(p * (n - 1))];
}
static long long c3_entries(void) {
    long long total = 0;
    char path[128];
    for (int cpu = 0; cpu < 256; cpu++) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpuidle/state3/usage", cpu);
        FILE *f = fopen(path, "r");
        if (!f) break;
        long long u;
        if (fscanf(f, "%lld", &u) == 1) total += u;
        fclose(f);
    }
    return total;
}

static double *buf[N_ITEMS];
static uint32_t *perm[N_ITEMS];
static int n_lines;

static double gather(int item, int passes) {  // latency-bound random walk over the item's cache lines
    double s = 0;
    double *b = buf[item];
    uint32_t *p = perm[item];
    for (int r = 0; r < passes; r++)
        for (int j = 0; j < n_lines; j++) s += b[(size_t)p[j] * 8];
    return s;
}

int main(int argc, char **argv) {
    int iters = atoi(argv[1]);
    int item_kb = atoi(argv[2]);
    int passes = atoi(argv[3]);
    int T = omp_get_max_threads();
    n_lines = item_kb * 1024 / 64;

    double t0 = now_us();
    spin(20000000);
    double k_per_us = 20000000 / (now_us() - t0);

    double *region = malloc(iters * sizeof(double));
    double *wake = malloc(iters * sizeof(double));
    double *slowest = malloc(iters * sizeof(double));
    double *median_busy = malloc(iters * sizeof(double));
    double arrive[MAX_T], busy[MAX_T], tmp[MAX_T];
    int cpu[MAX_T], prev_cpu[MAX_T];

    // First-touch each item's buffer from the thread that will own it.
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < N_ITEMS; i++) {
            buf[i] = aligned_alloc(64, (size_t)n_lines * 64);
            perm[i] = malloc(n_lines * sizeof(uint32_t));
            for (int j = 0; j < n_lines * 8; j++) buf[i][j] = j;
            for (int j = 0; j < n_lines; j++) perm[i][j] = j;
            unsigned seed = 1234 + i;
            for (int j = n_lines - 1; j > 0; j--) {
                int k = rand_r(&seed) % (j + 1);
                uint32_t t = perm[i][j]; perm[i][j] = perm[i][k]; perm[i][k] = t;
            }
        }
        prev_cpu[omp_get_thread_num()] = sched_getcpu();
    }

    printf("T=%d item_kb=%d passes=%d per_thread_kb=%d\n", T, item_kb, passes,
           ((N_ITEMS + T - 1) / T) * item_kb);
    printf("%7s %10s %10s %12s %14s %14s %10s %11s\n", "gap_us", "region_med", "region_p90",
           "wake_max_med", "busy_med_thr", "busy_max_thr", "migrated%", "C3_per_iter");
    for (int g = 4; g < argc; g++) {
        double gap_us = atof(argv[g]);
        long k_gap = (long)(gap_us * k_per_us);
        long long migrations = 0;
        long long c3_before = c3_entries();
        for (int it = -50; it < iters; it++) {  // 50 warm-up iterations
            spin(k_gap);
            double start = now_us();
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                arrive[tid] = now_us() - start;
                cpu[tid] = sched_getcpu();
                double b0 = now_us(), s = 0;
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < N_ITEMS; i++) s += gather(i, passes);
                busy[tid] = now_us() - b0;
                sink = s;
            }
            double end = now_us();
            if (it < 0) { for (int t = 0; t < T; t++) prev_cpu[t] = cpu[t]; continue; }
            region[it] = end - start;
            double w = 0, s = 0;
            for (int t = 0; t < T; t++) {
                if (t > 0 && arrive[t] > w) w = arrive[t];
                if (busy[t] > s) s = busy[t];
                tmp[t] = busy[t];
                migrations += cpu[t] != prev_cpu[t];
                prev_cpu[t] = cpu[t];
            }
            wake[it] = w; slowest[it] = s;
            median_busy[it] = pct(tmp, T, 0.5);
        }
        long long c3 = c3_entries() - c3_before;
        printf("%7.0f %10.1f %10.1f %12.1f %14.1f %14.1f %10.1f %11.1f\n", gap_us,
               pct(region, iters, 0.5), pct(region, iters, 0.9), pct(wake, iters, 0.5),
               pct(median_busy, iters, 0.5), pct(slowest, iters, 0.5),
               100.0 * migrations / ((double)iters * T), (double)c3 / iters);
        fflush(stdout);
    }
    return 0;
}
