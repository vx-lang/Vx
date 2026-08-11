//===- test_worker.h - A stand-in for a remote worker -----------*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The far side of a dispatch, for tests. Shared by the in-process round trip
// (loopback_test.cpp) and the two-process one (transport_test.cpp) so there is
// one worker rather than two that drift -- the same argument runtime/vx_agent.h
// makes about there being one dispatch decoder.
//
// Deliberately not a GPU. It resolves handles the way a real worker would and
// then runs the reference loop nest, so what the tests exercise is the
// plumbing rather than a vendor library. Its allocator is a bump arena, which
// is enough because nothing here frees.
//
//===----------------------------------------------------------------------===//

#ifndef VX_TEST_WORKER_H
#define VX_TEST_WORKER_H

#include "../../runtime/vx_agent.h"
#include "../../runtime/vx_dispatch_plan.h"

#include <cstring>

namespace vx_test {

// Its memory, its region table, and the arena a published result comes from.
vx_remote_region worker_regions[16];
vx_remote_table worker_table;
uint8_t worker_arena[8 * 1024 * 1024];
size_t worker_arena_used;

const uint32_t WORKER = 3;

void worker_reset() {
  memset(worker_regions, 0, sizeof(worker_regions));
  memset(worker_arena, 0, sizeof(worker_arena));
  worker_arena_used = 0;
  vx_remote_table_init(&worker_table, worker_regions,
                       sizeof(worker_regions) / sizeof(worker_regions[0]));
}

// Bump allocation, and the cursor moves only when the request is granted.
//
// It used to advance before the bounds check, so a single oversized request
// left `worker_arena_used` past the end of the arena and every later allocation
// failed too -- one refusal poisoning a worker that still had memory. That is
// the shape of bug an arena makes easy and a test only notices when something
// large comes through.
void *worker_alloc(size_t bytes) {
  size_t need = (bytes + 15) & ~(size_t)15;
  if (need > sizeof(worker_arena) - worker_arena_used) {
    return nullptr;
  }
  void *p = worker_arena + worker_arena_used;
  worker_arena_used += need;
  return p;
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

} // namespace vx_test

#endif /* VX_TEST_WORKER_H */
