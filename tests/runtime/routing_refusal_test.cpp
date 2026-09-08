//===- routing_refusal_test.cpp - Vx Compiler -------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What a backend does when it is handed an address belonging to another
// machine and the routing declined to serve it (#348).
//
// The local path is a memcpy or a free, and both are wrong for a handle: the
// address is non-canonical by construction, so the process dies on a signal
// with no output and a backtrace naming only the handler. That is what a fleet
// run produced, and reading it took a round trip to a rented machine -- the
// fault says nothing about manifests, topologies, or which worker holds the
// memory, all of which are known at the moment of the refusal.
//
// Two cases, because a guard that fires on everything is as bad as one that
// fires on nothing: an ordinary pointer has to pass through untouched, since
// every local run in the suite goes through these same entry points.
//
// Run as a subprocess by tests/integration_test/remote_client_test.rs, which
// checks the exit signal and the message -- this aborts on purpose.
//
//===----------------------------------------------------------------------===//

#include "vx_remote_routing.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s handle|pointer\n", argv[0]);
    return 2;
  }

  if (strcmp(argv[1], "handle") == 0) {
    // Worker 3, some offset. No manifest is set, so routing owns nothing and
    // every `try` declines -- which is exactly the state a mis-addressed
    // operation reaches.
    void *p = (void *)(uintptr_t)vx_remote_addr(3, 4096);
    vx_routing_refuse_handle("a read-back", p, 500);
    fprintf(stderr, "returned from a handle refusal\n");
    return 1;
  }

  // The ordinary case: real memory, no manifest, nothing to say.
  int x = 0;
  vx_routing_refuse_handle("a read-back", &x, 0);
  vx_routing_refuse_handle("a free", &x, 500);
  printf("passed through\n");
  return 0;
}
