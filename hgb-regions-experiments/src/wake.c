// Cost of an OpenMP parallel region that follows a serial busy phase of GAP us.
// Mimics HGB: serial find_node_split (the gap), then a histogram region of
// N_ITEMS features of ITEM_US each, statically scheduled.
// Usage: wake ITERS ITEM_US GAP_US...   (threads from OMP_NUM_THREADS)
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
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
static void work(long k) {  // fixed instruction count: slower clocks show up as longer work
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
static long long c3_entries(void) {  // sum over CPUs of the deepest idle state's usage counter
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

int main(int argc, char **argv) {
    int iters = atoi(argv[1]);
    double item_us = atof(argv[2]);
    int T = omp_get_max_threads();

    double t0 = now_us();
    work(20000000);
    double k_per_us = 20000000 / (now_us() - t0);
    long k_item = (long)(item_us * k_per_us);

    double *region = malloc(iters * sizeof(double));
    double *wake = malloc(iters * sizeof(double));
    double *slowest = malloc(iters * sizeof(double));
    int *wake_cpu = malloc(iters * sizeof(int));
    double arrive[MAX_T], busy[MAX_T];
    int cpu[MAX_T];

    #pragma omp parallel
    work(k_item);  // start the team

    printf("T=%d item_us=%.0f ideal_region_us=%.0f\n", T, item_us,
           ((N_ITEMS + T - 1) / T) * item_us);
    printf("%8s %10s %10s %12s %12s %14s %10s %12s\n", "gap_us", "region_med", "region_p90",
           "wake_max_med", "wake_max_p90", "slowest_busy_med", "slow_wake_E%", "C3_per_iter");
    for (int g = 3; g < argc; g++) {
        double gap_us = atof(argv[g]);
        long k_gap = (long)(gap_us * k_per_us);
        long long c3_before = c3_entries();
        for (int it = 0; it < iters; it++) {
            work(k_gap);  // serial phase on the main thread, other threads idle
            double start = now_us();
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                arrive[tid] = now_us() - start;
                cpu[tid] = sched_getcpu();
                double b0 = now_us();
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < N_ITEMS; i++) work(k_item);
                busy[tid] = now_us() - b0;
            }
            region[it] = now_us() - start;
            double w = 0, s = 0;
            int wc = -1;
            for (int t = 1; t < T; t++) {
                if (arrive[t] > w) { w = arrive[t]; wc = cpu[t]; }
                if (busy[t] > s) s = busy[t];
            }
            wake[it] = w; slowest[it] = s; wake_cpu[it] = wc;
        }
        long long c3 = c3_entries() - c3_before;
        int e = 0;
        for (int it = 0; it < iters; it++) e += wake_cpu[it] >= 8;  // cpu_atom = 8-23 on gamer
        printf("%8.0f %10.1f %10.1f %12.1f %12.1f %14.1f %10.0f %12.1f\n", gap_us,
               pct(region, iters, 0.5), pct(region, iters, 0.9), pct(wake, iters, 0.5),
               pct(wake, iters, 0.9), pct(slowest, iters, 0.5), 100.0 * e / iters,
               (double)c3 / iters);
        fflush(stdout);
    }
    return 0;
}
