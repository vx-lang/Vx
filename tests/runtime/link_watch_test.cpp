//===- link_watch_test.cpp - A host connection watches its link -*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A lost machine or link sends no EOF. So a host connection turns on TCP
// keepalive, and on Linux TCP_USER_TIMEOUT, with VX_REMOTE_LINK_TIMEOUT as
// the limit. A test cannot lose a link without root, so this test checks the
// options on the socket that vx_remote_connect opens.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>

static int failures = 0;

static void expect(int fd, int level, int name, int want, const char *what) {
  int got = -1;
  socklen_t len = sizeof(got);
  if (getsockopt(fd, level, name, &got, &len) != 0 || got != want) {
    fprintf(stderr, "FAIL: %s is %d, not %d\n", what, got, want);
    ++failures;
  }
}

// A flag that is on reads back as 1 on Linux, and as the value of the flag
// on macOS.
static void expect_on(int fd, int level, int name, const char *what) {
  int got = 0;
  socklen_t len = sizeof(got);
  if (getsockopt(fd, level, name, &got, &len) != 0 || got == 0) {
    fprintf(stderr, "FAIL: %s is off\n", what);
    ++failures;
  }
}

int main() {
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t len = sizeof(addr);
  if (listener < 0 || bind(listener, (sockaddr *)&addr, sizeof(addr)) != 0 ||
      listen(listener, 1) != 0 ||
      getsockname(listener, (sockaddr *)&addr, &len) != 0) {
    perror("listener");
    return 1;
  }

  // 12 s: 4 s before the first probe, then 4 probes 2 s apart.
  setenv("VX_REMOTE_LINK_TIMEOUT", "12", 1);
  vx_manifest m;
  vx_manifest_init(&m);
  vx_manifest_add(&m, "Worker", "127.0.0.1", ntohs(addr.sin_port));
  vx_remote_pool pool;
  vx_remote_pool_init(&pool);
  int fd = vx_remote_connect(&pool, vx_manifest_find(&m, "Worker"));
  if (fd < 0) {
    return 1;
  }

  expect_on(fd, SOL_SOCKET, SO_KEEPALIVE, "SO_KEEPALIVE");
#if defined(TCP_KEEPIDLE)
  expect(fd, IPPROTO_TCP, TCP_KEEPIDLE, 4, "TCP_KEEPIDLE");
#elif defined(TCP_KEEPALIVE)
  expect(fd, IPPROTO_TCP, TCP_KEEPALIVE, 4, "TCP_KEEPALIVE");
#endif
  expect(fd, IPPROTO_TCP, TCP_KEEPINTVL, 2, "TCP_KEEPINTVL");
  expect(fd, IPPROTO_TCP, TCP_KEEPCNT, 4, "TCP_KEEPCNT");
#if defined(TCP_USER_TIMEOUT)
  expect(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, 12000, "TCP_USER_TIMEOUT");
#endif

  vx_remote_pool_close(&pool);
  close(listener);
  if (failures == 0) {
    printf("the host connection watches its link\n");
  }
  return failures == 0 ? 0 : 1;
}
