//===- remote_client_test.cpp - Talking to a worker process -----*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A host driving a real `vx-worker` process over TCP, through the client header
// a plugin will use (#348). Stage two operands, dispatch a matmul, take the
// handle back, fetch the bytes, check the arithmetic, free.
//
// The worker under test is built against the *CPU* backend, which is the point
// rather than a compromise: the worker contains no vendor code, so what it is
// depends only on what it was linked against, and a CPU one makes the whole
// distributed path testable with no GPU anywhere.
//
// Two bugs were found by running this, both of the kind that leave no trace:
//
//   * The CPU backend decoded a dispatch plan only to log it and always called
//     the outlined kernel, so it and the CUDA backend disagreed about whether a
//     matmul is routable. A worker has no outlined kernel -- the artifact is
//     never shipped -- so it aborted on the first matmul.
//   * The client sent FREE without reading its acknowledgement, leaving it in
//     the stream for the next read. The "unknown handle is refused" check below
//     passed by consuming that stale ack rather than because anything was
//     refused.
//
// Driven by tests/integration_test/remote_client_test.rs, which builds the
// worker, starts it on a free port, and runs this against it.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"

#include <cstdio>
#include <cstring>

static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s host port\n", argv[0]);
    return 2;
  }

  vx_manifest m;
  vx_manifest_init(&m);
  check(vx_manifest_add(&m, "DecodeWorker", argv[1], atoi(argv[2])) == 1,
        "manifest entry added");

  const vx_manifest_entry *w = vx_manifest_find(&m, "DecodeWorker");
  check(w != nullptr, "DecodeWorker resolves");
  check(vx_manifest_find(&m, "SomewhereElse") == nullptr,
        "an unnamed worker stays local");

  vx_remote_pool pool;
  vx_remote_pool_init(&pool);
  int fd = vx_remote_connect(&pool, w);
  check(fd >= 0, "connected to the worker");
  if (fd < 0) {
    return 1;
  }

  static uint8_t scratch[1 << 20];
  const float A[4] = {1, 2, 3, 4};
  const float B[4] = {5, 6, 7, 8};
  const float expected[4] = {19, 22, 43, 50};

  uint64_t ha = vx_remote_transfer(fd, A, sizeof(A), VX_DTYPE_F32, scratch,
                                   sizeof(scratch));
  uint64_t hb = vx_remote_transfer(fd, B, sizeof(B), VX_DTYPE_F32, scratch,
                                   sizeof(scratch));
  check(ha != 0 && hb != 0, "both operands staged over TCP");
  check(ha != hb, "and got distinct handles");
  printf("  handle A = %llx\n  handle B = %llx\n", (unsigned long long)ha,
         (unsigned long long)hb);

  static const char payload[] = "vx_npu_kernel_0\0kind=matmul\0"
                                "roles=a:0,b:1,out:2\0outkind=slot\0"
                                "topo=1113\0toponame=DecodeWorker\0";
  vx_wire_arg args[3];
  memset(args, 0, sizeof(args));
  args[0].tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
  args[0].handle = ha;
  args[0].rank = 2;
  args[0].sizes[0] = args[0].sizes[1] = 2;
  args[0].strides[0] = 2;
  args[0].strides[1] = 1;
  args[1] = args[0];
  args[1].handle = hb;
  args[2].tag = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);

  vx_wire_writer w2;
  vx_wire_writer_init(&w2, scratch, sizeof(scratch));
  vx_wire_put_dispatch(&w2, payload, sizeof(payload), args, 3);
  check(vx_transport_send(fd, VX_WIRE_DISPATCH, scratch, w2.len) ==
            VX_TRANSPORT_OK,
        "dispatch sent");

  uint8_t reply[4096];
  uint32_t type = 0;
  uint64_t rlen = 0;
  check(vx_transport_recv(fd, &type, reply, sizeof(reply), &rlen) ==
            VX_TRANSPORT_OK,
        "reply received");

  vx_wire_reader rr;
  vx_wire_reader_init(&rr, reply, (size_t)rlen);
  int32_t status = -1;
  int64_t count = 0;
  check(vx_wire_get_results_header(&rr, &status, &count), "reply decodes");
  check(status == 0, "worker served the dispatch");
  check(count == 1, "one slot result");

  vx_wire_result res;
  if (count == 1 && vx_wire_get_result(&rr, &res)) {
    printf("  result handle = %llx, %lldx%lld\n",
           (unsigned long long)res.handle, (long long)res.sizes[0],
           (long long)res.sizes[1]);
    check(vx_remote_addr_is_handle(res.handle), "the result is a handle");

    float c[4] = {0, 0, 0, 0};
    check(vx_remote_fetch(fd, res.handle, c, sizeof(c)), "fetched the result");
    for (int i = 0; i < 4; ++i) {
      char what[64];
      snprintf(what, sizeof(what), "C[%d] == %g over TCP", i,
               (double)expected[i]);
      check(c[i] == expected[i], what);
    }
    printf("  C = [%g %g; %g %g]\n", (double)c[0], (double)c[1], (double)c[2],
           (double)c[3]);

    check(vx_remote_free(fd, res.handle), "freed the result");
  }

  // A fetch for a handle no worker minted must yield nothing.
  float junk[4] = {9, 9, 9, 9};
  check(!vx_remote_fetch(fd, vx_remote_addr(7, 1 << 20), junk, sizeof(junk)),
        "a fetch for an unknown handle is refused");

  vx_remote_pool_close(&pool);

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
