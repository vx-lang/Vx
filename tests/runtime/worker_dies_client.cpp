//===- worker_dies_client.cpp - A host whose worker dies --------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A host connects, prints "connected", and waits for a line on stdin. Then it
// sends a TRANSFER and waits for the reply. The test kills the worker in that
// wait. The TRANSFER must then fail, and this program exits with 3.
//
// usage: worker_dies_client <host> <port>
// Driven by tests/integration_test/remote_client_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"

#include <cstdio>

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s host port\n", argv[0]);
    return 2;
  }
  vx_manifest m;
  vx_manifest_init(&m);
  vx_manifest_add(&m, "Worker", argv[1], atoi(argv[2]));
  vx_remote_pool pool;
  vx_remote_pool_init(&pool);
  int fd = vx_remote_connect(&pool, vx_manifest_find(&m, "Worker"));
  if (fd < 0) {
    return 1;
  }
  printf("connected\n");
  fflush(stdout);
  char line[8];
  if (!fgets(line, sizeof line, stdin)) {
    return 1;
  }

  static uint8_t scratch[4096];
  const float a[4] = {1, 2, 3, 4};
  uint64_t h = vx_remote_transfer(fd, a, sizeof(a), VX_DTYPE_F32, scratch,
                                  sizeof(scratch));
  if (h != 0) {
    fprintf(stderr, "the TRANSFER got a reply from a worker that was killed\n");
    return 0;
  }
  fprintf(stderr, "the TRANSFER failed\n");
  return 3;
}
