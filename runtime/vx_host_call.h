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
#include <string.h> /* vx_host_worker_count_for: strchr */
#include <vector>

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h> /* vx_host_worker_count_for: sched_getaffinity */
#endif

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

/// Which worker is running this kernel, and how many there are.
///
/// A kernel the frontend proved has disjoint iterations is compiled so its
/// outermost loop covers only this worker's slice of the range. The compiler
/// emits calls to these rather than adding kernel parameters, so the C
/// interface of every outlined kernel keeps the shape its other callers already
/// know.
///
/// Thread-local, because one kernel is called from several threads at once and
/// each has to read its own index. A kernel that was never split reads them too
/// and gets 0 of 1, which is the whole range.
static __thread int64_t vx_host_worker_index = 0;
static __thread int64_t vx_host_worker_total = 1;

extern "C" int64_t vx_host_worker_id(void) { return vx_host_worker_index; }

extern "C" int64_t vx_host_worker_count(void) { return vx_host_worker_total; }

/// How many workers a dispatch should use, from what its payload claims.
///
/// Returns 1 -- run it serially, exactly as before -- unless the payload says
/// the kernel was split, and the split is the shape this path knows how to
/// drive. Getting that test wrong is the dangerous direction: running an
/// unsplit kernel on several threads makes every worker walk the WHOLE loop, so
/// the work is done N times and every captured write races with N-1 copies of
/// itself.
///
/// `launch=` is emitted for exactly the regions `parallel_outer_for` proved, so
/// its presence IS the proof travelling to the runtime. The comma form is the
/// two-level one (`launch=B,T`, Vx#379), whose host clone is not split -- what a
/// host worker should own across a block/thread mapping is a separate question
/// -- so it is declined here rather than half-driven.
static inline int vx_host_worker_count_for(const void *payload,
                                           size_t payload_size) {
  const char *launch = vx_payload_field(payload, payload_size, "launch=");
  if (!launch) {
    return 1;
  }
  long trip = strtol(launch, nullptr, 10);
  if (trip <= 1 || strchr(launch, ',')) {
    return 1;
  }

  // An explicit count wins, because it is how the effect of this gets measured
  // at all: the same binary, one variable. Zero or nonsense means serial.
  if (const char *env = getenv("VX_HOST_THREADS")) {
    long want = strtol(env, nullptr, 10);
    if (want < 1) {
      return 1;
    }
    return (int)(want < trip ? want : trip);
  }

  // Otherwise every CPU this thread is allowed to run on. After a placed
  // dispatch that is one NUMA node's CPUs, because the caller has already
  // pinned itself to the node holding the arguments and the workers inherit
  // that mask -- so the pool cannot spill onto the other socket and pay for the
  // interconnect on every access. Unplaced, it is the whole machine.
  long cpus = 1;
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    cpus = CPU_COUNT(&set);
  }
#else
  cpus = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  if (cpus < 1) {
    cpus = 1;
  }
  // More workers than iterations would only hand the extras an empty range.
  return (int)(cpus < trip ? cpus : trip);
}

/// One worker's call.
struct vx_host_worker_job {
  ffi_cif *cif;
  void *kernel;
  void **args;
  int64_t index;
  int64_t total;
};

static void *vx_host_worker_main(void *p) {
  vx_host_worker_job *job = static_cast<vx_host_worker_job *>(p);
  vx_host_worker_index = job->index;
  vx_host_worker_total = job->total;
  ffi_call(job->cif, reinterpret_cast<void (*)(void)>(job->kernel), nullptr,
           job->args);
  return nullptr;
}

/// Call the kernel with the platform's calling convention reconstructed from
/// the tags. The C-interface returns void; results flow through the memref
/// captures. Returns false only if libffi rejects the signature.
///
/// `workers` is how many disjoint slices to run at once, from
/// `vx_host_worker_count_for`. At 1 this is one `ffi_call` on the calling
/// thread, which is what this function has always been.
///
/// Threads are created per dispatch rather than kept in a standing pool. That
/// costs roughly 20us per worker, which is nothing against a kernel that reads
/// a GiB and would be most of the time for a kernel that reads a row. A
/// standing pool is the answer when a measurement says so; it is not the answer
/// before one, because it buys latency with a set of live threads, a shutdown
/// path, and a story about fork.
static inline bool vx_host_call_kernel(void *kernel, void **device_args,
                                       const int32_t *arg_tags,
                                       int64_t num_args, int workers = 1) {
  std::vector<ffi_type *> types(num_args > 0 ? (size_t)num_args : 1);
  for (int64_t i = 0; i < num_args; ++i) {
    types[i] = vx_abi_ffi_type(arg_tags[i]);
  }

  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, (unsigned)num_args, &ffi_type_void,
                   types.data()) != FFI_OK) {
    return false;
  }

  if (workers <= 1) {
    ffi_call(&cif, reinterpret_cast<void (*)(void)>(kernel), nullptr,
             device_args);
    return true;
  }

  // `ffi_call` on an already-prepared cif does not write to it, so one cif
  // serves every worker. Only `ffi_prep_cif` above is unsafe to share, and it
  // has already run.
  std::vector<vx_host_worker_job> jobs((size_t)workers);
  std::vector<pthread_t> threads((size_t)workers);
  std::vector<bool> started((size_t)workers, false);
  for (int w = 0; w < workers; ++w) {
    jobs[(size_t)w] = {&cif, kernel, device_args, w, workers};
  }

  // Worker 0 runs here, so the calling thread is not idle and keeps whatever
  // placement it was given.
  for (int w = 1; w < workers; ++w) {
    if (pthread_create(&threads[(size_t)w], nullptr, vx_host_worker_main,
                       &jobs[(size_t)w]) == 0) {
      started[(size_t)w] = true;
    } else {
      // Out of threads. Run this worker's slice here instead of skipping it --
      // a slice nobody runs is a hole in the output, and the iterations it
      // owned are exactly the ones no other worker will touch.
      vx_host_worker_main(&jobs[(size_t)w]);
    }
  }
  vx_host_worker_main(&jobs[0]);
  for (int w = 1; w < workers; ++w) {
    if (started[(size_t)w]) {
      pthread_join(threads[(size_t)w], nullptr);
    }
  }
  // Leave the calling thread as a single worker again, so anything that runs a
  // kernel without going through this function sees the whole range.
  vx_host_worker_index = 0;
  vx_host_worker_total = 1;
  return true;
}

#endif /* VX_HOST_CALL_H */
