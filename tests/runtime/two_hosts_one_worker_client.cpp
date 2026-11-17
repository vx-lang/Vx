//===- two_hosts_one_worker_client.cpp - Two hosts, one worker --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two connections to one worker act as two hosts, A and B. Each host sends its
// own data and reads it back. Then host A leaves, and the worker frees what A
// left. The data of host B must stay. The test tells this program on stdin
// when the worker logged that it freed the buffers of A.
//
// usage: two_hosts_one_worker_client <host> <port>
// Driven by tests/integration_test/remote_client_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>

static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

static int connect_host(const char *host, int port) {
  vx_manifest m;
  vx_manifest_init(&m);
  vx_manifest_add(&m, "Worker", host, port);
  vx_remote_pool pool;
  vx_remote_pool_init(&pool);
  return vx_remote_connect(&pool, vx_manifest_find(&m, "Worker"));
}

static bool holds(int fd, uint64_t handle, const float *want) {
  float got[4] = {0, 0, 0, 0};
  return vx_remote_fetch(fd, handle, got, sizeof(got)) &&
         memcmp(got, want, sizeof(got)) == 0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s host port\n", argv[0]);
    return 2;
  }
  int a = connect_host(argv[1], atoi(argv[2]));
  int b = connect_host(argv[1], atoi(argv[2]));
  if (a < 0 || b < 0) {
    return 1;
  }

  static uint8_t scratch[4096];
  const float data_a[4] = {1, 2, 3, 4};
  const float data_b[4] = {5, 6, 7, 8};
  uint64_t ha = vx_remote_transfer(a, data_a, sizeof(data_a), VX_DTYPE_F32,
                                   scratch, sizeof(scratch));
  uint64_t hb = vx_remote_transfer(b, data_b, sizeof(data_b), VX_DTYPE_F32,
                                   scratch, sizeof(scratch));
  check(ha != 0 && hb != 0, "each host sends its data");
  check(holds(a, ha, data_a), "host A reads back its own data");
  check(holds(b, hb, data_b), "host B reads back its own data");

  // The worker frees the buffers of A when it sees that A left. The test reads
  // the log of the worker, and sends a line when the worker did that.
  close(a);
  printf("A left\n");
  fflush(stdout);
  char line[8];
  if (!fgets(line, sizeof line, stdin)) {
    return 1;
  }
  check(holds(b, hb, data_b), "host B keeps its data after host A leaves");

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  return 0;
}
