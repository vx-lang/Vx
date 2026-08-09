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
// non-CPU `spawn on` has no definition for vx_plugin_dispatch_async and fails to
// link — which is the state Linux was in before this file existed (build.rs
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

#include "../include/vx_hardware_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <dlfcn.h>
#include <ffi.h>

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

// Map a Vx ABI type tag (abiTagForType in src/dialect/VxLowering.cpp) to a
// libffi type. Keep this switch in sync with the producer's encoding and with
// the copy in runtime/npu_dispatch.mm.
ffi_type *abi_ffi_type(int32_t tag) {
  switch (tag) {
  case 1:
    return &ffi_type_uint8; // i1
  case 2:
    return &ffi_type_sint8; // i8
  case 3:
    return &ffi_type_sint16; // i16
  case 4:
    return &ffi_type_sint32; // i32
  case 5:
    return &ffi_type_sint64; // i64
  case 6:
    return &ffi_type_float; // f32
  case 7:
    return &ffi_type_double; // f64
  case 0:
  default:
    return &ffi_type_pointer; // memref descriptor / fallback
  }
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
  (void)payload_size;
  const char *kernel_name = static_cast<const char *>(binary_payload);

  if (verbose()) {
    fprintf(stderr, "[Vx Dispatcher] host execution of %s (%lld args)\n",
            kernel_name, static_cast<long long>(num_args));
  }

  char ciface_name[256];
  snprintf(ciface_name, sizeof(ciface_name), "_mlir_ciface_%s", kernel_name);

  void *kernel = dlsym(RTLD_DEFAULT, ciface_name);
  if (!kernel) {
    fprintf(stderr, "[Vx Dispatcher] FATAL: outlined kernel %s not found\n",
            ciface_name);
    abort();
  }

  std::vector<ffi_type *> types(num_args > 0 ? static_cast<size_t>(num_args)
                                             : 1);
  for (int64_t i = 0; i < num_args; ++i) {
    types[i] = abi_ffi_type(arg_tags[i]);
  }

  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, static_cast<unsigned>(num_args),
                   &ffi_type_void, types.data()) != FFI_OK) {
    fprintf(stderr, "[Vx Dispatcher] FATAL: ffi_prep_cif failed for %s\n",
            ciface_name);
    abort();
  }

  // The kernel C-interface returns void; results flow through memref captures.
  ffi_call(&cif, reinterpret_cast<void (*)(void)>(kernel), nullptr, device_args);
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
