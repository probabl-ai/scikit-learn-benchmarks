// Does a barrier at the start of a parallel loop change where the threads run, and the region time?
// usage: barrier_placement BARRIER(0|1) THREADS WORK_ITERS GAP_ITERS REGIONS
// Each thread does the same work per region, with a serial gap between regions, as for HGB's nodes.
// Stacked is the number of threads sharing a CPU with another thread when they start the loop.
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CPUS 512

static volatile double sink;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void work(long n) {
    double x = 0;
    for (long i = 0; i < n; i++) x += i * 1e-9;
    sink = x;
}

int main(int argc, char **argv) {
    int barrier = atoi(argv[1]), T = atoi(argv[2]);
    long n = atol(argv[3]), gap = atol(argv[4]);
    int regions = atoi(argv[5]);
    int *cpu_start = calloc(T, sizeof(int)), *cpu_end = calloc(T, sizeof(int));
    double *t_loop = calloc(T, sizeof(double)), *t_entry = calloc(T, sizeof(double));
    double wall = 0, loop_max = 0, loop_mean = 0, entry_max = 0;
    long stacked = 0, regions_stacked = 0, moved = 0;

    for (int r = 0; r < regions; r++) {
        double t0 = now();
#pragma omp parallel num_threads(T)
        {
            int me = omp_get_thread_num();
            if (barrier) {
#pragma omp barrier
            }
            cpu_start[me] = sched_getcpu();
            double ts = now();
            t_entry[me] = ts - t0;
#pragma omp for nowait schedule(static)
            for (int i = 0; i < T; i++) work(n);
            t_loop[me] = now() - ts;
            cpu_end[me] = sched_getcpu();
        }
        wall += now() - t0;

        int count[MAX_CPUS];
        memset(count, 0, sizeof(count));
        double lmax = 0, emax = 0;
        for (int i = 0; i < T; i++) {
            count[cpu_start[i]]++;
            moved += cpu_start[i] != cpu_end[i];
            loop_mean += t_loop[i] / T;
            if (t_loop[i] > lmax) lmax = t_loop[i];
            if (t_entry[i] > emax) emax = t_entry[i];
        }
        long here = 0;
        for (int c = 0; c < MAX_CPUS; c++) if (count[c] > 1) here += count[c];
        stacked += here;
        regions_stacked += here > 0;
        loop_max += lmax;
        entry_max += emax;
        work(gap);
    }
    printf("barrier=%d T=%d n=%ld wall_us=%.1f loop_mean_us=%.1f loop_max_us=%.1f entry_max_us=%.1f "
           "stacked_threads=%.2f regions_stacked=%.3f moved_threads=%.3f\n",
           barrier, T, n, 1e6 * wall / regions, 1e6 * loop_mean / regions, 1e6 * loop_max / regions,
           1e6 * entry_max / regions, (double)stacked / regions, (double)regions_stacked / regions,
           (double)moved / regions);
    return 0;
}
