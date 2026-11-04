// One SCHED_IDLE thread pinned on each CPU, sleeping on a periodic timer. The menu idle governor bounds its
// prediction by the next timer, so a period below a state's target residency keeps the CPU out of that state
// without root. The thread sleeps nearly all the time, so unlike a spinner the CPU stays free for wake-up placement.
// usage: keepawake PERIOD_US NCPUS   (runs until SIGTERM, then prints the wake-ups per CPU)
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <time.h>

static long period_ns;
static long counts[1024];

static void *loop(void *arg) {
    long cpu = (long)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) {
        perror("affinity");
        exit(1);
    }
    struct sched_param param = {0};
    if (sched_setscheduler(0, SCHED_IDLE, &param)) {
        perror("sched_setscheduler");
        exit(1);
    }
    // The default 50 us slack would let the kernel merge and delay these timers.
    prctl(PR_SET_TIMERSLACK, 1UL);
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    for (;;) {
        next.tv_nsec += period_ns;
        while (next.tv_nsec >= 1000000000L) {
            next.tv_nsec -= 1000000000L;
            next.tv_sec++;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
        // Absolute deadlines: when the thread ran late, restart from now instead of firing a burst to catch up.
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec > next.tv_sec || (now.tv_sec == next.tv_sec && now.tv_nsec > next.tv_nsec + period_ns))
            next = now;
        __atomic_add_fetch(&counts[cpu], 1, __ATOMIC_RELAXED);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    period_ns = atol(argv[1]) * 1000;
    long ncpus = atol(argv[2]);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &set, NULL);
    for (long cpu = 0; cpu < ncpus; cpu++) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, loop, (void *)cpu))
            return 1;
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int sig;
    sigwait(&set, &sig);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double seconds = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    long total = 0;
    for (long cpu = 0; cpu < ncpus; cpu++)
        total += counts[cpu];
    printf("# keepawake period_us=%s seconds=%.1f wakes_per_cpu_per_s=%.0f\n", argv[1], seconds, total / seconds / ncpus);
    fflush(stdout);
    return 0;
}
