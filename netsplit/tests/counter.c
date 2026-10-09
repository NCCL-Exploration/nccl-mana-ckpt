/*
 * counter: the smallest netsplit smoke test (no CUDA, no NCCL).  Counts for
 * -T seconds plus 5 s after the last pause (a checkpoint), keeps a heap
 * buffer and a thread busy, and checks at the end that both the buffer and
 * argv survived: under netsplit-lh this exercises the loader, the upper
 * half's memory arena, the region filter and the restart path.
 *
 *   counter SECONDS "an argument with spaces"
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile long g_ticks;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *ticker(void *arg) {
    (void)arg;
    for (;;) {
        g_ticks++;
        usleep(1000);
    }
    return NULL;
}

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 10;
    const char *arg2 = argc > 2 ? argv[2] : "";
    char *copy = strdup(arg2);
    size_t n = 64 << 20;
    unsigned char *buf = malloc(n);
    for (size_t i = 0; i < n; i += 4096)
        buf[i] = (unsigned char)(i / 4096);
    pthread_t th;
    pthread_create(&th, NULL, ticker, NULL);

    double t0 = now(), prev = t0, resumed = 0, last = 0;
    long it = 0;
    for (;; it++) {
        double t = now();
        if (t - prev > 2) {
            resumed = t;
            printf("PAUSE %.1f s\n", t - prev);
        }
        prev = t;
        if (t - last > 1) {
            last = t;
            printf("PROGRESS counter iter %ld ticks %ld\n", it, g_ticks);
            fflush(stdout);
        }
        if (t - t0 > seconds && t - resumed > 5)
            break;
        usleep(20000);
    }
    for (size_t i = 0; i < n; i += 4096)
        if (buf[i] != (unsigned char)(i / 4096)) {
            printf("RESULT counter: FAIL heap byte %zu\n", i);
            return 1;
        }
    if (strcmp(arg2, copy) != 0) {
        printf("RESULT counter: FAIL argv '%s' was '%s'\n", arg2, copy);
        return 1;
    }
    printf("RESULT counter: PASS iters=%ld ticks=%ld argv='%s'\n", it, g_ticks, arg2);
    return 0;
}
