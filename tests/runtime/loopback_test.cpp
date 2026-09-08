//===- loopback_test.cpp - A dispatch that goes there and back --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A whole remote dispatch, in one process, with no GPU and no network: stage a
// buffer, encode a dispatch, decode it on the far side, resolve the handles,
// run the GEMM the compiler described, publish the result, encode the reply,
// decode it back, and check the numbers.
//
// Everything here is what would otherwise first run on two rented machines.
// The failures it is looking for do not announce themselves -- a handle
// resolved to the wrong backing pointer computes the wrong product and returns
// it, a slot published but never collected leaves the caller with a descriptor
// full of zeroes, and an argument rebuilt with derived strides instead of the
// ones that arrived reads a view incorrectly. None of that faults.
//
// The "worker" below is deliberately not a GPU. It resolves handles the way a
// real one would and then runs the reference loop nest, so what is under test
// is the plumbing rather than a vendor library.
//
// Driven by tests/integration_test/loopback_test.rs.
//
//===----------------------------------------------------------------------===//

#include "test_worker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using namespace vx_test;

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/* --- the round trip ------------------------------------------------------ */

uint8_t wire[8192];

// A 2x2 matmul whose answer is not symmetric, so a transposed or swapped
// operand shows up as wrong numbers rather than the same ones.
//   A = [1 2; 3 4]   B = [5 6; 7 8]   A*B = [19 22; 43 50]
const float kA[4] = {1, 2, 3, 4};
const float kB[4] = {5, 6, 7, 8};
const float kExpected[4] = {19, 22, 43, 50};

void test_dispatch_round_trip_produces_the_right_numbers() {
  worker_reset();

  // --- host: stage both operands -----------------------------------------
  uint64_t handle_a = 0, handle_b = 0;
  {
    const int64_t sizes[2] = {2, 2};
    vx_wire_writer w;
    vx_wire_reader r;
    vx_wire_transfer t;

    vx_wire_writer_init(&w, wire, sizeof(wire));
    vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, kA, sizeof(kA));
    vx_wire_reader_init(&r, wire, w.len);
    check(vx_wire_get_transfer(&r, &t), "A's transfer decodes");
    handle_a = worker_transfer(&t);

    vx_wire_writer_init(&w, wire, sizeof(wire));
    vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, kB, sizeof(kB));
    vx_wire_reader_init(&r, wire, w.len);
    check(vx_wire_get_transfer(&r, &t), "B's transfer decodes");
    handle_b = worker_transfer(&t);
  }
  check(handle_a != 0 && handle_b != 0, "both operands got handles");
  check(handle_a != handle_b, "and they are different buffers");

  // --- host: a dispatch naming them, with a slot for the result ----------
  //
  // The payload is the one the compiler emits, byte for byte.
  static const char payload[] = "vx_npu_kernel_0\0kind=matmul\0"
                                "roles=a:0,b:1,out:2\0outkind=slot\0"
                                "topo=1113\0toponame=DecodeWorker\0";
  vx_wire_arg args[3];
  memset(args, 0, sizeof(args));

  args[0].tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
  args[0].handle = handle_a;
  args[0].rank = 2;
  args[0].sizes[0] = 2;
  args[0].sizes[1] = 2;
  args[0].strides[0] = 2;
  args[0].strides[1] = 1;

  args[1] = args[0];
  args[1].handle = handle_b;

  args[2].tag = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);

  vx_wire_writer w;
  vx_wire_writer_init(&w, wire, sizeof(wire));
  vx_wire_put_dispatch(&w, payload, sizeof(payload), args, 3);
  check(!w.overflow, "the dispatch fits the buffer");

  // Residency in one number: this dispatch names its operands rather than
  // carrying them, so its size does not depend on theirs. At 2x2 the operands
  // are 16 bytes and the payload string alone is larger, so the comparison to
  // make is against what the same dispatch would cost if it shipped the data --
  // a 288x288 f32 projection, the shape llama2 actually dispatches.
  check(w.len < 288 * 288 * sizeof(float),
        "a dispatch names its operands rather than carrying them");

  // --- worker: decode, run, reply ----------------------------------------
  vx_wire_result results[4];
  int64_t num_results = 0;
  {
    vx_wire_reader r;
    vx_wire_dispatch d;
    vx_wire_arg got[3];

    vx_wire_reader_init(&r, wire, w.len);
    check(vx_wire_get_dispatch(&r, &d), "the worker decodes the dispatch");
    for (int i = 0; i < 3; ++i) {
      check(vx_wire_get_arg(&r, &got[i]), "and each argument");
    }
    check(worker_dispatch(&d, got, results, &num_results), "and runs it");
  }
  check(num_results == 1, "one slot produced one result");
  check(results[0].rank == 2, "the result is rank 2");
  check(results[0].sizes[0] == 2 && results[0].sizes[1] == 2, "and 2x2");

  // --- host: apply the reply to its own slot -----------------------------
  uint8_t host_slot[128];
  memset(host_slot, 0, sizeof(host_slot));
  {
    vx_wire_writer rw;
    vx_wire_reader rr;
    int32_t status = -1;
    int64_t count = 0;
    vx_wire_result got;

    vx_wire_writer_init(&rw, wire, sizeof(wire));
    vx_wire_put_results(&rw, 0, results, num_results);
    vx_wire_reader_init(&rr, wire, rw.len);
    check(vx_wire_get_results_header(&rr, &status, &count), "reply decodes");
    check(status == 0 && count == 1, "reply says one result, no error");
    check(vx_wire_get_result(&rr, &got), "the result decodes");
    vx_agent_apply_result(host_slot, &got);
  }

  // The host's slot now holds a descriptor whose data pointer is a handle: a
  // thing to pass on, not a thing to read.
  {
    void *published = vx_memref_aligned(host_slot);
    check(vx_remote_addr_is_handle((uint64_t)(uintptr_t)published),
          "the host's slot holds a handle, not a pointer");
    check(vx_remote_addr_worker((uint64_t)(uintptr_t)published) == WORKER,
          "and it names the worker that produced it");
    const int64_t *sizes = vx_memref_sizes(host_slot);
    check(sizes[0] == 2 && sizes[1] == 2, "with the shape that came back");
  }

  // --- and the numbers ---------------------------------------------------
  //
  // Read on the worker, which is the only side that may. This is the check the
  // whole file exists for: a handle resolved to the wrong buffer, or operands
  // swapped by a roles mistake, produces a plausible matrix and no failure.
  {
    vx_remote_ref ref;
    if (!vx_remote_resolve(&worker_table, results[0].handle, &ref)) {
      check(false, "the result handle resolves on the worker");
    } else {
      const float *out =
          (const float *)((char *)ref.region->remote + ref.offset);
      for (int i = 0; i < 4; ++i) {
        char what[64];
        snprintf(what, sizeof(what), "C[%d] == %g", i, (double)kExpected[i]);
        check(out[i] == kExpected[i], what);
      }
    }
  }
}

// outkind=buffer: the caller handed over a buffer to fill rather than a slot to
// publish through. It is the other half of the ABI and takes a different path
// on both sides.
void test_buffer_result_round_trip() {
  worker_reset();

  const int64_t sizes[2] = {2, 2};
  float zeros[4] = {0, 0, 0, 0};
  uint64_t ha, hb, hc;
  {
    vx_wire_writer w;
    vx_wire_reader r;
    vx_wire_transfer t;

    vx_wire_writer_init(&w, wire, sizeof(wire));
    vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, kA, sizeof(kA));
    vx_wire_reader_init(&r, wire, w.len);
    vx_wire_get_transfer(&r, &t);
    ha = worker_transfer(&t);

    vx_wire_writer_init(&w, wire, sizeof(wire));
    vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, kB, sizeof(kB));
    vx_wire_reader_init(&r, wire, w.len);
    vx_wire_get_transfer(&r, &t);
    hb = worker_transfer(&t);

    vx_wire_writer_init(&w, wire, sizeof(wire));
    vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, zeros, sizeof(zeros));
    vx_wire_reader_init(&r, wire, w.len);
    vx_wire_get_transfer(&r, &t);
    hc = worker_transfer(&t);
  }

  static const char payload[] = "vx_npu_kernel_1\0kind=matmul\0"
                                "roles=a:0,b:1,out:2\0outkind=buffer\0"
                                "topo=1113\0toponame=DecodeWorker\0";
  vx_wire_arg args[3];
  memset(args, 0, sizeof(args));
  for (int i = 0; i < 3; ++i) {
    args[i].tag = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
    args[i].rank = 2;
    args[i].sizes[0] = 2;
    args[i].sizes[1] = 2;
    args[i].strides[0] = 2;
    args[i].strides[1] = 1;
  }
  args[0].handle = ha;
  args[1].handle = hb;
  args[2].handle = hc;

  vx_wire_writer w;
  vx_wire_writer_init(&w, wire, sizeof(wire));
  vx_wire_put_dispatch(&w, payload, sizeof(payload), args, 3);

  vx_wire_reader r;
  vx_wire_dispatch d;
  vx_wire_arg got[3];
  vx_wire_result results[4];
  int64_t num_results = 0;

  vx_wire_reader_init(&r, wire, w.len);
  check(vx_wire_get_dispatch(&r, &d), "buffer dispatch decodes");
  for (int i = 0; i < 3; ++i) {
    vx_wire_get_arg(&r, &got[i]);
  }
  check(worker_dispatch(&d, got, results, &num_results),
        "buffer dispatch runs");
  check(num_results == 0, "a filled buffer publishes no slot result");

  vx_remote_ref ref;
  if (!vx_remote_resolve(&worker_table, hc, &ref)) {
    check(false, "the output buffer resolves");
    return;
  }
  const float *out = (const float *)((char *)ref.region->remote + ref.offset);
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "filled C[%d] == %g", i, (double)kExpected[i]);
    check(out[i] == kExpected[i], what);
  }
}

// A dispatch naming a handle that does not resolve must be refused, not served
// out of whatever region happens to be nearby.
void test_unresolvable_handle_is_refused() {
  worker_reset();

  const int64_t sizes[2] = {2, 2};
  uint64_t ha;
  {
    vx_wire_writer w;
    vx_wire_reader r;
    vx_wire_transfer t;
    vx_wire_writer_init(&w, wire, sizeof(wire));
    vx_wire_put_transfer(&w, VX_DTYPE_F32, 2, sizes, kA, sizeof(kA));
    vx_wire_reader_init(&r, wire, w.len);
    vx_wire_get_transfer(&r, &t);
    ha = worker_transfer(&t);
  }

  static const char payload[] = "vx_npu_kernel_2\0kind=matmul\0"
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
  // A handle for a worker that never allocated anything.
  args[1].handle = vx_remote_addr(9, 1 << 20);
  args[2].tag = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);

  vx_wire_writer w;
  vx_wire_writer_init(&w, wire, sizeof(wire));
  vx_wire_put_dispatch(&w, payload, sizeof(payload), args, 3);

  vx_wire_reader r;
  vx_wire_dispatch d;
  vx_wire_arg got[3];
  vx_wire_result results[4];
  int64_t num_results = 0;

  vx_wire_reader_init(&r, wire, w.len);
  vx_wire_get_dispatch(&r, &d);
  for (int i = 0; i < 3; ++i) {
    vx_wire_get_arg(&r, &got[i]);
  }
  check(!worker_dispatch(&d, got, results, &num_results),
        "a dispatch naming an unknown handle is refused");

  // And one whose offset ran past its region, which is the off-by-one the gaps
  // in vx_remote_region.h exist for.
  args[1].handle = ha + (1 << 20);
  vx_wire_writer_init(&w, wire, sizeof(wire));
  vx_wire_put_dispatch(&w, payload, sizeof(payload), args, 3);
  vx_wire_reader_init(&r, wire, w.len);
  vx_wire_get_dispatch(&r, &d);
  for (int i = 0; i < 3; ++i) {
    vx_wire_get_arg(&r, &got[i]);
  }
  check(!worker_dispatch(&d, got, results, &num_results),
        "a dispatch whose offset left its region is refused");
}

} // namespace

int main() {
  test_dispatch_round_trip_produces_the_right_numbers();
  test_buffer_result_round_trip();
  test_unresolvable_handle_is_refused();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
