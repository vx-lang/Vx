//===- vx_host_call.h - Run an outlined kernel on the host ------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Calling an outlined kernel through its MLIR C-interface, given the untyped
// argument array the plugin ABI hands over.
//
// Every backend needs this. It is what the portable shim does for all kernels,
// and what the accelerator backends do for the ones they do not recognise --
// a plugin that cannot route a kernel must still run it. The ABI-tag to libffi
// mapping had been copied into each backend with a comment asking that the
// copies be kept in sync, which is the arrangement that stops being true
// quietly: a tag added in one place and missed in another misreads an argument
// rather than failing to build.
//
//===----------------------------------------------------------------------===//

#ifndef VX_HOST_CALL_H
#define VX_HOST_CALL_H

#include "../include/vx_hardware_runtime.h"

#include <stdio.h>
#include <stdlib.h> /* vx_host_refuse_coop: abort */
#include <vector>

#include <dlfcn.h>

// Homebrew's libffi is not on the default include path on macOS, where the SDK
// instead provides the header under a directory of its own.
#if defined(__APPLE__)
#include <ffi/ffi.h>
#else
#include <ffi.h>
#endif

/// libffi type for an ABI tag. Only the kind byte participates: a memref's
/// element type and rank ride in the high bytes and do not change how the
/// argument is passed.
static inline ffi_type *vx_abi_ffi_type(int32_t tag) {
  switch (VX_ABI_KIND(tag)) {
  case VX_ABI_KIND_I1:
    return &ffi_type_uint8;
  case VX_ABI_KIND_I8:
    return &ffi_type_sint8;
  case VX_ABI_KIND_I16:
    return &ffi_type_sint16;
  case VX_ABI_KIND_I32:
    return &ffi_type_sint32;
  case VX_ABI_KIND_I64:
    return &ffi_type_sint64;
  case VX_ABI_KIND_F32:
    return &ffi_type_float;
  case VX_ABI_KIND_F64:
    return &ffi_type_double;
  case VX_ABI_KIND_MEMREF:
  default:
    return &ffi_type_pointer; /* memref descriptor / fallback */
  }
}

/// Refuse the host schedule of a cooperative kernel, loudly.
///
/// A payload carrying `coop=` names a kernel whose barriers are INSIDE its
/// thread loop (Vx#379 stage C). Such a kernel has no serial schedule: running
/// the loop nest in program order walks one thread's whole body -- every tile,
/// every barrier -- before the next thread ever stages its rows, and the
/// consume phases read rows nobody filled. That is not slow, it is wrong, and
/// it is wrong silently: the numbers come out plausible. So every backend calls
/// this before running an outlined kernel on the host, and a cooperative kernel
/// dies with its reason instead of lying.
static inline void vx_host_refuse_coop(const void *payload,
                                       size_t payload_size,
                                       const char *kernel_name,
                                       const char *backend) {
  if (!vx_payload_field(payload, payload_size, "coop=")) {
    return;
  }
  fprintf(stderr,
          "[Vx %s] FATAL: %s is a cooperative kernel -- its barriers are\n"
          "        inside the thread loop, so no serial order of that loop\n"
          "        computes it. It needs a device; the host has no schedule\n"
          "        for it.\n",
          backend, kernel_name);
  abort();
}

/// Resolve an outlined kernel's C-interface entry point. Returns NULL when the
/// symbol is absent; what to do about that is the caller's policy, which
/// differs between backends.
static inline void *vx_host_kernel_symbol(const char *kernel_name) {
  char ciface_name[256];
  snprintf(ciface_name, sizeof(ciface_name), "_mlir_ciface_%s", kernel_name);
  return dlsym(RTLD_DEFAULT, ciface_name);
}

/// Call the kernel with the platform's calling convention reconstructed from
/// the tags. The C-interface returns void; results flow through the memref
/// captures. Returns false only if libffi rejects the signature.
static inline bool vx_host_call_kernel(void *kernel, void **device_args,
                                       const int32_t *arg_tags,
                                       int64_t num_args) {
  std::vector<ffi_type *> types(num_args > 0 ? (size_t)num_args : 1);
  for (int64_t i = 0; i < num_args; ++i) {
    types[i] = vx_abi_ffi_type(arg_tags[i]);
  }

  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, (unsigned)num_args, &ffi_type_void,
                   types.data()) != FFI_OK) {
    return false;
  }

  ffi_call(&cif, reinterpret_cast<void (*)(void)>(kernel), nullptr,
           device_args);
  return true;
}

#endif /* VX_HOST_CALL_H */
