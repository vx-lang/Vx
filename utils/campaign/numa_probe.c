// numa_probe.c - measure memory bandwidth for one (cpu node, memory node) pair.
//
// Run under numactl, which decides both ends:
//
//   numactl --cpunodebind=0 --membind=1 ./numa_probe read 2048
//
// The placement is numactl's job, not this program's. What this program has to
// get right is that the pages are actually faulted in on the bound node before
// the timed loop starts -- Linux allocates on first touch, so an untouched
// buffer would be placed by whichever access came first inside the timing.
//
// Two kernels, because they stress the link differently:
//
//   read - sum every element. One byte of traffic per byte of buffer, and the
//          cleanest measure of how fast a node reaches memory.
//   copy - memcpy between two buffers. Two bytes of traffic per byte copied
//          (a read and a write), so its byte rate cannot exceed half the peak
//          even in principle. Reported as traffic, not as copied bytes.
//
// Prints one JSON object so the caller does not have to parse prose.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Touch every page so the kernel commits it on the node numactl bound us to.
// Done in parallel with the same schedule the timed loop uses, so each thread
// faults the pages it will later read: under --membind the node is fixed either
// way, but this keeps the per-thread locality the same in both phases.
static void first_touch(char *p, size_t bytes) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < bytes; i += 4096) {
    p[i] = 1;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: numa_probe <read|copy> <MiB> [reps]\n");
    return 2;
  }
  const char *kernel = argv[1];
  size_t mib = (size_t)strtoull(argv[2], NULL, 10);
  int reps = argc > 3 ? atoi(argv[3]) : 5;
  size_t bytes = mib * 1024 * 1024;

  int is_copy = strcmp(kernel, "copy") == 0;
  if (!is_copy && strcmp(kernel, "read") != 0) {
    fprintf(stderr, "unknown kernel '%s'\n", kernel);
    return 2;
  }

  char *a = aligned_alloc(4096, bytes);
  char *b = is_copy ? aligned_alloc(4096, bytes) : NULL;
  if (!a || (is_copy && !b)) {
    fprintf(stderr, "out of memory for %zu MiB\n", mib);
    return 1;
  }
  first_touch(a, bytes);
  if (is_copy) {
    first_touch(b, bytes);
  }

  int threads = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp master
    threads = omp_get_num_threads();
  }
#endif

  double best = 1e30, total = 0.0;
  volatile uint64_t sink = 0;

  for (int r = 0; r < reps; r++) {
    double t0 = now_s();
    if (is_copy) {
      memcpy(b, a, bytes);
    } else {
      uint64_t sum = 0;
      const uint64_t *p = (const uint64_t *)a;
      size_t n = bytes / sizeof(uint64_t);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : sum)
#endif
      for (size_t i = 0; i < n; i++) {
        sum += p[i];
      }
      sink = sum;
    }
    double dt = now_s() - t0;
    if (dt < best) {
      best = dt;
    }
    total += dt;
  }
  (void)sink;

  // Traffic, not buffer size: a copy moves the bytes twice.
  double traffic = (double)bytes * (is_copy ? 2.0 : 1.0);
  double best_gbs = traffic / best / 1e9;
  double mean_gbs = traffic / (total / reps) / 1e9;

  printf("{\"kernel\":\"%s\",\"mib\":%zu,\"reps\":%d,\"threads\":%d,"
         "\"best_s\":%.6f,\"mean_s\":%.6f,"
         "\"best_gbs\":%.2f,\"mean_gbs\":%.2f}\n",
         kernel, mib, reps, threads, best, total / reps, best_gbs, mean_gbs);

  free(a);
  free(b);
  return 0;
}
