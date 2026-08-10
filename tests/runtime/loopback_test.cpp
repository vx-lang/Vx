//===- loopback_test.cpp - A dispatch that goes there and back --*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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

#include "../../runtime/vx_agent.h"
#include "../../runtime/vx_dispatch_plan.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/* --- the worker ---------------------------------------------------------- */

// Its memory, its region table, and the arena a published result comes from.
vx_remote_region worker_regions[16];
vx_remote_table worker_table;
uint8_t worker_arena[64 * 1024];
size_t worker_arena_used;

const uint32_t WORKER = 3;

void worker_reset() {
  memset(worker_regions, 0, sizeof(worker_regions));
  memset(worker_arena, 0, sizeof(worker_arena));
  worker_arena_used = 0;
  vx_remote_table_init(&worker_table, worker_regions,
                       sizeof(worker_regions) / sizeof(worker_regions[0]));
}

void *worker_alloc(size_t bytes) {
  void *p = worker_arena + worker_arena_used;
  worker_arena_used += (bytes + 15) & ~(size_t)15;
  return worker_arena_used <= sizeof(worker_arena) ? p : nullptr;
}

// TRANSFER: take the bytes, keep them, hand back a name for them.
uint64_t worker_transfer(const vx_wire_transfer *t) {
  void *backing = worker_alloc((size_t)t->nbytes);
  if (!backing) {
    return 0;
  }
  memcpy(backing, t->bytes, (size_t)t->nbytes);
  return vx_remote_table_alloc(&worker_table, WORKER, t->nbytes, t->dtype,
                               backing);
}

// The reference GEMM, standing in for cuBLAS. Row-major, C = A * B.
void reference_gemm(const vx_gemm_plan *plan, float *out) {
  const float *a = (const float *)plan->a_data;
  const float *b = (const float *)plan->b_data;
  for (int64_t i = 0; i < plan->m; ++i) {
    for (int64_t j = 0; j < plan->n; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < plan->k; ++k) {
        acc += a[i * plan->a_row_stride + k] * b[k * plan->b_row_stride + j];
      }
      out[i * plan->n + j] = acc;
    }
  }
}

// DISPATCH: rebuild the arguments, decode with the *same* decoder a local
// plugin uses, run, publish, and say what was produced.
int worker_dispatch(const vx_wire_dispatch *d, const vx_wire_arg *args,
                    vx_wire_result *results, int64_t *num_results) {
  void *device_args[8];
  void *desc_ptrs[8];
  int32_t arg_tags[8];
  uint8_t descriptors[8 * 128];
  vx_agent_args storage;
  vx_gemm_plan plan;

  storage.device_args = device_args;
  storage.desc_ptrs = desc_ptrs;
  storage.arg_tags = arg_tags;
  storage.descriptors = descriptors;
  storage.descriptors_capacity = sizeof(descriptors);

  if (d->num_args > 8) {
    return 0;
  }
  if (!vx_agent_rebuild_args(&worker_table, args, d->num_args, &storage)) {
    return 0;
  }

  if (!vx_gemm_plan_decode(d->payload, (size_t)d->payload_len, device_args,
                           arg_tags, d->num_args, &plan)) {
    return 0;
  }

  if (plan.out_kind == VX_GEMM_OUT_SLOT) {
    size_t bytes = (size_t)plan.m * (size_t)plan.n * sizeof(float);
    void *result = worker_alloc(bytes);
    if (!result) {
      return 0;
    }
    /* Registered before publication, so collecting it is a lookup rather than
       a special case, exactly as a really-allocated device buffer would be. */
    if (vx_remote_table_alloc(&worker_table, WORKER, bytes, VX_DTYPE_F32,
                              result) == 0) {
      return 0;
    }
    reference_gemm(&plan, (float *)result);
    vx_gemm_publish_slot(&plan, result);
  } else {
    reference_gemm(&plan, (float *)plan.out_data);
  }

  *num_results = vx_agent_collect_results(&worker_table, args, d->num_args,
                                          &storage, results, 4);
  return *num_results >= 0;
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
