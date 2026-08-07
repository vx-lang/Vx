// measure_link.c -- achieved copy bandwidth across a size sweep (vx-review#13, M1 protocol).
//
// The measurement half of the shakedown. Emits one CSV row per (size, rep-median) so the compare
// step joins it against the compiler's predictions by byte count.
//
// Protocol, mirroring what the M-series will do on rented hardware, so that the *protocol* is what
// gets debugged here rather than on a metered box:
//
//   * log-spaced sizes from 4 KiB to 256 MiB, both to catch a latency floor at the small end and
//     to get past every cache at the large end;
//   * >= 9 reps, median reported with IQR, because a mean over a noisy laptop is not a measurement;
//   * a warm-up rep per size, discarded -- the first touch pays page faults, and on a lazily-mapped
//     allocation that is a first-touch cost, not a bandwidth;
//   * buffers re-touched between reps so the copy is not reading from a cache it just filled;
//   * `clock_gettime(CLOCK_MONOTONIC)`, never wall time.
//
// Reports BOTH the copy rate (bytes copied / time) and the traffic rate (2x that, since a memcpy
// reads a byte and writes a byte). Which of the two a declared "memory bandwidth" figure should be
// compared against is exactly the sort of question a dry run is supposed to force into the open --
// so the harness reports both rather than picking one and hiding the choice.
//
//   cc -O2 measure_link.c -o measure_link && ./measure_link > raw.csv
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef REPS
#define REPS 11
#endif

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(void) {
  // 4 KiB .. 256 MiB, log-spaced (x2). The small end is where a bandwidth-only model is expected
  // to under-predict cost because a latency floor dominates; measuring that crossover IS a result.
  size_t sizes[64];
  int n = 0;
  for (size_t s = 4096; s <= (256ul << 20); s *= 2) sizes[n++] = s;

  printf("bytes,median_ns,q1_ns,q3_ns,copy_GBps,traffic_GBps,reps\n");

  for (int i = 0; i < n; ++i) {
    size_t bytes = sizes[i];
    char *src = malloc(bytes);
    char *dst = malloc(bytes);
    if (!src || !dst) {
      fprintf(stderr, "alloc failed at %zu bytes\n", bytes);
      free(src); free(dst);
      continue;
    }
    // First touch every page so the timed region measures copying, not faulting.
    memset(src, 0x5a, bytes);
    memset(dst, 0x00, bytes);

    memcpy(dst, src, bytes);  // warm-up, discarded

    // How many copies per timed region. `CLOCK_MONOTONIC` on this machine has ~1 us granularity,
    // so a single 4 KiB copy times as either 0 or 1000 ns -- which yields an infinite bandwidth
    // and, worse, looks like a number. Timing a calibrated batch and dividing is the fix, and
    // finding this on the laptop instead of on a rented box is the whole point of the dry run.
    //
    // The batch is grown until the timed region is >= TARGET_NS, so the clock's granularity is a
    // negligible fraction of what is being measured at every size.
    const double TARGET_NS = 5e6;  // 5 ms
    long iters = 1;
    for (;;) {
      double t0 = now_sec();
      for (long k = 0; k < iters; ++k) memcpy(dst, src, bytes);
      double elapsed = (now_sec() - t0) * 1e9;
      if (elapsed >= TARGET_NS || iters >= (1L << 30)) break;
      // Scale straight to the target rather than doubling, with a floor so a zero reading (the
      // very case this loop exists for) still makes progress.
      long next = elapsed > 0 ? (long)(iters * TARGET_NS / elapsed) + 1 : iters * 8;
      iters = next > iters ? next : iters * 2;
    }

    double samples[REPS];
    for (int r = 0; r < REPS; ++r) {
      double t0 = now_sec();
      for (long k = 0; k < iters; ++k) memcpy(dst, src, bytes);
      double t1 = now_sec();
      samples[r] = (t1 - t0) * 1e9 / (double)iters;  // ns per copy
    }
    // Keep the result observable so the copy cannot be optimised away.
    volatile char sink = dst[bytes - 1];
    (void)sink;

    qsort(samples, REPS, sizeof(double), cmp_double);
    double med = samples[REPS / 2];
    double q1 = samples[REPS / 4];
    double q3 = samples[(3 * REPS) / 4];
    if (med <= 0.0) {
      // Still unmeasurable. Report it as missing rather than as an infinite bandwidth: a row that
      // says nothing is recoverable, a row that says `inf` gets averaged into a result.
      fprintf(stderr, "%zu bytes: unmeasurable even at %ld iters/rep -- row skipped\n", bytes,
              iters);
      free(src);
      free(dst);
      continue;
    }
    double copy_gbps = (double)bytes / med;              // bytes/ns == GB/s
    double traffic_gbps = 2.0 * copy_gbps;               // a memcpy reads AND writes

    printf("%zu,%.1f,%.1f,%.1f,%.2f,%.2f,%d\n", bytes, med, q1, q3, copy_gbps, traffic_gbps, REPS);
    fflush(stdout);

    free(src);
    free(dst);
  }
  return 0;
}
