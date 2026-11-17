//===- two_workers_client.cpp - A host that uses two workers ----*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A host that puts a buffer on its first worker, and then uses its second
// worker while it keeps the first connection open. The test runs two of these
// hosts with the two workers in opposite order. Each host waits until the other
// put its first buffer, so each holds a connection to the worker that the other
// asks next.
//
// Usage: two_workers_client HOST FIRST_PORT SECOND_PORT MY_FILE OTHER_FILE
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>

static uint8_t scratch[1 << 16];

/// Puts `data` on the worker and reads it back. Returns the handle, or 0.
static uint64_t put_and_check(int fd, const float *data, const char *which) {
  uint64_t h = vx_remote_transfer(fd, data, 4 * sizeof(float), VX_DTYPE_F32,
                                  scratch, sizeof(scratch));
  float back[4] = {0, 0, 0, 0};
  if (h == 0 || !vx_remote_fetch(fd, h, back, sizeof(back)) ||
      memcmp(back, data, sizeof(back)) != 0) {
    fprintf(stderr, "FAIL: the buffer on the %s worker did not come back\n",
            which);
    return 0;
  }
  return h;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr,
            "usage: %s HOST FIRST_PORT SECOND_PORT MY_FILE OTHER_FILE\n",
            argv[0]);
    return 2;
  }
  vx_manifest m;
  vx_manifest_init(&m);
  vx_manifest_add(&m, "First", argv[1], atoi(argv[2]));
  vx_manifest_add(&m, "Second", argv[1], atoi(argv[3]));
  vx_remote_pool pool;
  vx_remote_pool_init(&pool);

  const float a[4] = {1, 2, 3, 4};
  const float b[4] = {5, 6, 7, 8};
  int first = vx_remote_connect(&pool, vx_manifest_find(&m, "First"));
  if (first < 0 || put_and_check(first, a, "first") == 0) {
    return 1;
  }

  // Tell the other host, and wait until it holds its first worker too.
  FILE *mine = fopen(argv[4], "w");
  if (!mine) {
    return 1;
  }
  fclose(mine);
  for (int i = 0; access(argv[5], F_OK) != 0; ++i) {
    if (i == 1000) {
      fprintf(stderr, "FAIL: the other host did not start\n");
      return 1;
    }
    usleep(10000);
  }

  int second = vx_remote_connect(&pool, vx_manifest_find(&m, "Second"));
  if (second < 0 || put_and_check(second, b, "second") == 0) {
    return 1;
  }
  // The first connection still works after the second one.
  if (put_and_check(first, b, "first") == 0) {
    return 1;
  }
  vx_remote_pool_close(&pool);
  printf("both workers answered\n");
  return 0;
}
