//===- host_dispatch.cpp - Portable Vx dispatch shim ------------*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Provider of the vx_plugin_* ABI (include/vx_hardware_runtime.h) for platforms
// with no accelerator backend compiled in. Without it, a program containing a
// non-CPU `spawn on` has no definition for vx_plugin_dispatch_async and fails
// to link — which is the state Linux was in before this file existed (build.rs
// compiled the dispatcher only on macOS).
//
// Execution semantics: kernels outlined by VxLowering run on the host CPU
// through their MLIR C-interface, called via libffi so that by-value floats and
// small integers land in the registers the platform ABI requires. Device memory
// is host memory. Placement, staging and capacity remain compile-time facts; on
// this path they carry no runtime effect.
//
// The Apple backend (runtime/npu_dispatch.mm) implements the same ABI with real
// ANE/AMX dispatch and falls back to this same libffi path. A CUDA sibling
// (#321 M1) specializes the device-memory operations rather than replacing the
// fallback.
//
//===----------------------------------------------------------------------===//

#include "vx_host_call.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Diagnostics are opt-in: the backend test harness compares a program's stdout
// against its `// EXPECT:` lines, so an unconditional banner would fail every
// placed test on this path.
bool verbose() {
  static const bool on = [] {
    const char *v = getenv("VX_DISPATCH_VERBOSE");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

// Describe one argument on stderr, for VX_DISPATCH_VERBOSE. The point is to
// show what a plugin has to work with: the tag says what each pointer is, and
// for anything ranked the descriptor says how big it is.
void describe_arg(int64_t i, int32_t tag, void *arg) {
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

  // A slot holds a descriptor rather than elements, and the tag describes the
  // one it holds -- so the shape below is read one indirection further in.
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

void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id) {
  (void)topology_id;
  void *ptr = malloc(bytes);
  if (ptr && host_ptr) {
    memcpy(ptr, host_ptr, bytes);
  }
  return ptr;
}

uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                  size_t payload_size, void **device_args,
                                  const int32_t *arg_tags, int64_t num_args) {
  const char *kernel_name = static_cast<const char *>(binary_payload);

  if (verbose()) {
    const char *kind = vx_payload_field(binary_payload, payload_size, "kind=");
    const char *roles =
        vx_payload_field(binary_payload, payload_size, "roles=");
    const char *outkind =
        vx_payload_field(binary_payload, payload_size, "outkind=");
    fprintf(stderr,
            "[Vx Dispatcher] host execution of %s (%lld args), kind=%s "
            "roles=%s outkind=%s\n",
            kernel_name, static_cast<long long>(num_args),
            kind ? kind : "<unclassified>", roles ? roles : "-",
            outkind ? outkind : "-");
    for (int64_t i = 0; i < num_args; ++i) {
      describe_arg(i, arg_tags[i], device_args[i]);
    }
  }

  void *kernel = vx_host_kernel_symbol(kernel_name);
  if (!kernel) {
    fprintf(stderr, "[Vx Dispatcher] FATAL: outlined kernel %s not found\n",
            kernel_name);
    abort();
  }

  if (!vx_host_call_kernel(kernel, device_args, arg_tags, num_args)) {
    fprintf(stderr, "[Vx Dispatcher] FATAL: could not build a call for %s\n",
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
                                          size_t bytes) {
  memcpy(host_ptr, device_ptr, bytes);
  return 1;
}

void vx_plugin_free(void *device_ptr, uint32_t topology_id) {
  (void)topology_id;
  free(device_ptr);
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
