//===- heap_count.c - Vx Compiler -------------------------------*- C -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Counts the heap blocks a compiled Vx program allocates and frees, for `fuzz.py --heap`.
//
// Preloaded with LD_PRELOAD into `vxc`, which runs the compiled program as `temp.out`. Only
// that process counts, so the compiler and the linker are left out. When it exits, it appends
// "<allocations> <frees>" to the file VX_HEAP_COUNT_FILE names. Linux and glibc only: it
// calls glibc's own allocator underneath.
//
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
extern void __libc_free(void *);

static int counting;
static long allocations, frees;

static void count(long *counter) {
  if (counting)
    __atomic_add_fetch(counter, 1, __ATOMIC_RELAXED);
}

__attribute__((constructor)) static void start(void) {
  char exe[4096];
  ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (n <= 0)
    return;
  exe[n] = 0;
  const char *name = "/temp.out";
  size_t len = strlen(name);
  counting = (size_t)n >= len && strcmp(exe + n - len, name) == 0;
}

__attribute__((destructor)) static void stop(void) {
  const char *path = getenv("VX_HEAP_COUNT_FILE");
  if (!counting || !path)
    return;
  char line[64];
  int len = snprintf(line, sizeof line, "%ld %ld\n", allocations, frees);
  int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0)
    return;
  if (write(fd, line, len) != len)
    perror("heap_count");
  close(fd);
}

void *malloc(size_t n) {
  void *p = __libc_malloc(n);
  if (p)
    count(&allocations);
  return p;
}

void *calloc(size_t n, size_t size) {
  void *p = __libc_calloc(n, size);
  if (p)
    count(&allocations);
  return p;
}

void *realloc(void *old, size_t n) {
  void *p = __libc_realloc(old, n);
  if (!old && p)
    count(&allocations);
  if (old && !n)
    count(&frees);
  return p;
}

void *aligned_alloc(size_t alignment, size_t n) {
  void *p = __libc_memalign(alignment, n);
  if (p)
    count(&allocations);
  return p;
}

int posix_memalign(void **out, size_t alignment, size_t n) {
  void *p = __libc_memalign(alignment, n);
  if (!p)
    return 12; // ENOMEM
  *out = p;
  count(&allocations);
  return 0;
}

void free(void *p) {
  if (p)
    count(&frees);
  __libc_free(p);
}
