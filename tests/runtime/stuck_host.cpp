//===- stuck_host.cpp - A host that stops inside a message ------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A host that connects to a worker and then stops in one of two ways:
//
//   half     sends the first half of a request header, and nothing more.
//   no-read  puts a 32 MB buffer on the worker, asks for it back with a FETCH,
//            and does not read the reply. The reply is larger than the socket
//            buffers, so the worker cannot finish sending it.
//
// It then prints "stuck" and waits until the test kills it.
//
// Usage: stuck_host HOST PORT half|no-read
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s HOST PORT half|no-read\n", argv[0]);
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

  if (strcmp(argv[3], "half") == 0) {
    uint8_t header[VX_TRANSPORT_HEADER_BYTES];
    vx_wire_writer w;
    vx_wire_writer_init(&w, header, sizeof(header));
    vx_wire_put_header(&w, VX_WIRE_FETCH, 16);
    if (vx_transport_write_all(fd, header, sizeof(header) / 2) !=
        VX_TRANSPORT_OK) {
      return 1;
    }
  } else if (strcmp(argv[3], "no-read") == 0) {
    const uint64_t bytes = 32u << 20;
    uint8_t *data = (uint8_t *)calloc(bytes, 1);
    size_t scratch_len = bytes + 4096;
    uint8_t *scratch = (uint8_t *)malloc(scratch_len);
    uint64_t h =
        vx_remote_transfer(fd, data, bytes, VX_DTYPE_F32, scratch, scratch_len);
    if (h == 0) {
      fprintf(stderr, "FAIL: the worker did not take the buffer\n");
      return 1;
    }
    uint8_t request[32];
    vx_wire_writer w;
    vx_wire_writer_init(&w, request, sizeof(request));
    vx_wire_put_fetch(&w, h, bytes);
    if (vx_transport_send(fd, VX_WIRE_FETCH, request, w.len) !=
        VX_TRANSPORT_OK) {
      return 1;
    }
  } else {
    fprintf(stderr, "unknown mode %s\n", argv[3]);
    return 2;
  }

  printf("stuck\n");
  fflush(stdout);
  for (;;) {
    pause();
  }
}
