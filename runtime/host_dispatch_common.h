//===- host_dispatch_common.h - Shared CPU backend body ---------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The whole vx_plugin_* ABI for a CPU target, parameterized by the two things
// that actually differ between them: what the backend calls itself, and the
// alignment its vector unit wants.
//
// Every target has a backend. That is the point of the arrangement rather than
// an accident of it: the compiler emits calls to the plugin ABI -- allocate,
// transfer, free, dispatch -- and never names a vendor. `transfer` becomes
// cudaMalloc plus an H2D copy where runtime/cuda_dispatch.cpp is linked, and an
// aligned host allocation here, and the program is correct either way. A
// compiler that special-cased libc for "the CPU case" would have to know which
// case it was in, which is exactly the knowledge the plugin boundary exists to
// avoid.
//
// A backend including this file defines:
//   VX_BACKEND_NAME    what it calls itself in diagnostics
//   VX_BACKEND_ALIGN   allocation alignment in bytes
//
//===----------------------------------------------------------------------===//

#ifndef VX_HOST_DISPATCH_COMMON_H
#define VX_HOST_DISPATCH_COMMON_H

#include "vx_dispatch_plan.h"
#include "vx_host_call.h"
#include "vx_remote_routing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef VX_BACKEND_NAME
#error "a backend including host_dispatch_common.h must define VX_BACKEND_NAME"
#endif
#ifndef VX_BACKEND_ALIGN
#error "a backend including host_dispatch_common.h must define VX_BACKEND_ALIGN"
#endif

namespace {

// Diagnostics are opt-in: the backend test harness compares a program's stdout
// against its `// EXPECT:` lines, so an unconditional banner would fail every
// placed test.
inline bool vx_verbose() {
  static const bool on = [] {
    const char *v = getenv("VX_DISPATCH_VERBOSE");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

/// C = A * B, row-major, for the element types this backend can carry.
///
/// The reference implementation, deliberately: it exists so a CPU backend gives
/// the same *answer to the routing question* an accelerator one does, not so it
/// goes faster. See the call site in vx_plugin_dispatch_async.
template <typename T>
inline void vx_host_gemm_typed(const vx_gemm_plan &plan, T *out,
                               int64_t out_row_stride) {
  const T *a = static_cast<const T *>(plan.a_data);
  const T *b = static_cast<const T *>(plan.b_data);
  for (int64_t i = 0; i < plan.m; ++i) {
    for (int64_t j = 0; j < plan.n; ++j) {
      T acc = static_cast<T>(0);
      for (int64_t k = 0; k < plan.k; ++k) {
        acc += a[i * plan.a_row_stride + k] * b[k * plan.b_row_stride + j];
      }
      out[i * out_row_stride + j] = acc;
    }
  }
}

/// Run a decoded plan. Returns false for anything it cannot carry, which sends
/// the caller back to the outlined kernel -- a refusal costs performance and
/// never correctness, exactly as it does in the CUDA backend.
inline bool vx_host_run_gemm(const vx_gemm_plan &plan) {
  size_t esz = vx_dtype_bytes(plan.dtype);
  if (esz == 0 || (plan.dtype != VX_DTYPE_F32 && plan.dtype != VX_DTYPE_F64)) {
    return false;
  }

  void *out = plan.out_data;
  int64_t stride = plan.out_row_stride;

  if (plan.out_kind == VX_GEMM_OUT_SLOT) {
    /* Standing in for the kernel means doing what it would have: allocate the
       result and publish a descriptor naming it. The allocation outlives this
       call because the caller loads that descriptor out of the slot. */
    out = malloc((size_t)plan.m * (size_t)plan.n * esz);
    if (!out) {
      return false;
    }
    stride = plan.n;
  } else if (!out) {
    return false;
  }

  if (plan.dtype == VX_DTYPE_F32) {
    vx_host_gemm_typed<float>(plan, static_cast<float *>(out), stride);
  } else {
    vx_host_gemm_typed<double>(plan, static_cast<double *>(out), stride);
  }

  if (plan.out_kind == VX_GEMM_OUT_SLOT) {
    vx_gemm_publish_slot(&plan, out);
  }
  if (vx_verbose()) {
    fprintf(stderr, "[Vx " VX_BACKEND_NAME "] GEMM %lldx%lldx%lld %s -> %s\n",
            (long long)plan.m, (long long)plan.n, (long long)plan.k,
            vx_dtype_name(plan.dtype),
            plan.out_kind == VX_GEMM_OUT_SLOT ? "slot" : "buffer");
  }
  return true;
}

/// Describe one argument, for VX_DISPATCH_VERBOSE. The tag says what each
/// pointer is; for anything ranked the descriptor says how big it is.
inline void vx_describe_arg(int64_t i, int32_t tag, void *arg) {
  if (VX_ABI_KIND(tag) != VX_ABI_KIND_MEMREF) {
    fprintf(stderr, "  arg %lld: scalar kind=%d\n", (long long)i,
            VX_ABI_KIND(tag));
    return;
  }

  int32_t rank = VX_ABI_RANK(tag);
  int32_t elem = VX_ABI_ELEM(tag);
  if (rank == 0 && elem == VX_DTYPE_UNKNOWN) {
    // Kind 0 covers both a ranked memref and the pointer fallback in
    // abiTagForType; only the latter carries no element type or rank.
    fprintf(stderr, "  arg %lld: opaque ptr\n", (long long)i);
    return;
  }

  const void *desc = *(const void **)arg;
  bool is_slot = VX_ABI_IS_SLOT(tag);
  if (is_slot && desc) {
    desc = vx_memref_aligned(desc);
  }

  fprintf(stderr, "  arg %lld: %s %s rank=%d shape=[", (long long)i,
          is_slot ? "slot ->" : "memref", vx_dtype_name(elem), rank);
  if (desc) {
    const int64_t *sizes = vx_memref_sizes(desc);
    for (int32_t d = 0; d < rank; ++d) {
      fprintf(stderr, "%s%lld", d ? "x" : "", (long long)sizes[d]);
    }
  }
  fprintf(stderr, "] elem_bytes=%zu\n", vx_dtype_bytes(elem));
}

} // namespace

extern "C" {

/// Allocate in this backend's memory and copy into it.
///
/// A CPU backend's "device memory" is host memory, so this is an aligned
/// allocation -- but it goes through the same entry point a GPU backend
/// implements with cudaMalloc, which is what lets `transfer` mean the same
/// thing in a program compiled for either.
void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id) {
  void *remote = nullptr;
  if (vx_routing_try_alloc(bytes, host_ptr, topology_id, &remote)) {
    return remote;
  }
  if (bytes == 0) {
    return nullptr;
  }

  // Rounded up because aligned_alloc requires a size that is a multiple of the
  // alignment; the caller only ever addresses the bytes it asked for.
  size_t rounded =
      (bytes + VX_BACKEND_ALIGN - 1) & ~(size_t)(VX_BACKEND_ALIGN - 1);
  void *ptr = aligned_alloc(VX_BACKEND_ALIGN, rounded);
  if (!ptr) {
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] FATAL: out of memory for %zu bytes\n",
            bytes);
    abort();
  }
  if (host_ptr) {
    memcpy(ptr, host_ptr, bytes);
  }
  if (vx_verbose()) {
    fprintf(stderr, "[Vx " VX_BACKEND_NAME "] staged %zu bytes\n", bytes);
  }
  return ptr;
}

/// Release what the entry point above returned.
///
/// Paired with it deliberately: an allocation made by a backend has to be freed
/// by the same backend. Handing this pointer to libc would be correct here and
/// heap corruption under the CUDA backend, which is why the compiler emits this
/// call rather than a `free`.
void vx_plugin_free(void *device_ptr, uint32_t topology_id) {
  if (vx_routing_try_free(device_ptr, topology_id)) {
    return;
  }
  vx_routing_refuse_handle("a free", device_ptr, topology_id);
  free(device_ptr);
}

uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                  size_t payload_size, void **device_args,
                                  const int32_t *arg_tags, int64_t num_args) {
  const char *kernel_name = static_cast<const char *>(binary_payload);

  if (vx_routing_try_dispatch(binary_payload, payload_size, device_args,
                              arg_tags, num_args)) {
    return 1;
  }

  if (vx_verbose()) {
    const char *kind = vx_payload_field(binary_payload, payload_size, "kind=");
    const char *roles =
        vx_payload_field(binary_payload, payload_size, "roles=");
    const char *outkind =
        vx_payload_field(binary_payload, payload_size, "outkind=");
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] host execution of %s (%lld args), "
            "kind=%s roles=%s outkind=%s\n",
            kernel_name, static_cast<long long>(num_args),
            kind ? kind : "<unclassified>", roles ? roles : "-",
            outkind ? outkind : "-");
    for (int64_t i = 0; i < num_args; ++i) {
      vx_describe_arg(i, arg_tags[i], device_args[i]);
    }
  }

  /* A classified matmul runs here rather than through the outlined kernel, the
     same way the CUDA backend hands one to cuBLAS.
   *
   * This is not an optimisation -- the loop nest computes the same numbers, and
   * a reference GEMM is no faster. It is what makes this backend answer the
   * same question the accelerator ones do. Without it the two disagree about
   * whether a kernel is *routable*, and that difference is load-bearing in one
   * place: a remote worker (runtime/vx_worker_main.cpp) has no outlined kernel
   * to fall back to, because the artifact was never shipped to it. A CPU worker
   * that could not route a matmul would abort on the first one, which is
   * exactly what it did before this existed.
   *
   * So a CPU machine can serve dispatches, and a disaggregated program can be
   * developed and tested without a GPU anywhere -- the property the rest of
   this
   * ABI already has. */
  {
    vx_gemm_plan plan;
    if (vx_gemm_plan_decode(binary_payload, payload_size, device_args, arg_tags,
                            num_args, &plan) &&
        vx_host_run_gemm(plan)) {
      return 1;
    }
  }

  vx_host_refuse_coop(binary_payload, payload_size, kernel_name,
                      VX_BACKEND_NAME);
  void *kernel = vx_host_kernel_symbol(kernel_name);
  if (!kernel) {
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] FATAL: outlined kernel %s not found\n",
            kernel_name);
    abort();
  }

  if (!vx_host_call_kernel(kernel, device_args, arg_tags, num_args)) {
    fprintf(stderr,
            "[Vx " VX_BACKEND_NAME "] FATAL: could not build a call for %s\n",
            kernel_name);
    abort();
  }
  return 1;
}

uint64_t vx_plugin_dispatch_async_flat(float *xout, float *x, float *w, int n,
                                       int d) {
  // Reference matmul: xout[i] = sum_j w[i * n + j] * x[j], matching the
  // signature the flat path emits (see #31).
  for (int i = 0; i < d; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      acc += w[i * n + j] * x[j];
    }
    xout[i] = acc;
  }
  return 1;
}

void vx_plugin_await_future(uint64_t future_id) {
  // Dispatch is synchronous on this path, so the result is ready on return.
  (void)future_id;
}

int32_t vx_plugin_transfer_device_to_host(void *device_ptr, void *host_ptr,
                                          size_t bytes, uint32_t topology_id) {
  if (vx_routing_try_fetch(device_ptr, host_ptr, bytes, topology_id)) {
    return 1;
  }
  vx_routing_refuse_handle("a read-back", device_ptr, topology_id);
  // Otherwise one memory, so the topology names it and nothing follows.
  memcpy(host_ptr, device_ptr, bytes);
  return 1;
}

void *vx_plugin_transfer_peer(void *src_device_ptr, uint32_t src_topology_id,
                              uint32_t dst_topology_id, size_t bytes) {
  void *routed = nullptr;
  if (vx_routing_try_peer(src_device_ptr, src_topology_id, dst_topology_id,
                          bytes, &routed)) {
    return routed;
  }
  // A host backend has one memory, so both topologies name it and the movement
  // between them is a copy. Answering rather than refusing is what lets a
  // disaggregated program be developed and tested on a laptop: the same source
  // that moves a KV cache between two GPUs runs here, and produces the same
  // tokens, which is how the two-device run gets an oracle to be checked
  // against (#347).
  vx_routing_refuse_handle("a peer handoff", src_device_ptr, src_topology_id);
  (void)src_topology_id;
  (void)dst_topology_id;
  void *dst = malloc(bytes);
  if (dst && src_device_ptr) {
    memcpy(dst, src_device_ptr, bytes);
  }
  return dst;
}

void vx_plugin_release_future(uint64_t future_id) { (void)future_id; }

int32_t vx_plugin_control(uint32_t opcode, void *payload) {
  (void)payload;
  if (opcode == VX_CTRL_GET_DEVICE_COUNT) {
    // The host itself, modelled as one device.
    return 1;
  }
  return 0;
}

} // extern "C"

#endif /* VX_HOST_DISPATCH_COMMON_H */
