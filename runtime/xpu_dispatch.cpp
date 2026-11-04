//===- xpu_dispatch.cpp - Intel XPU backend for the Vx ABI --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provider of the vx_plugin_* ABI (include/vx_hardware_runtime.h) for Intel
// GPUs. A placed kernel compiled for `arch: spirv64` travels in the dispatch
// payload; this library takes the image out of the payload's section, loads it,
// launches it with the flattened argument list, and routes `kind=matmul` to
// oneMKL (#1137).
//
// Which runtime, and why two of them. The plan routes this backend through the
// SYCL *runtime*, because oneMKL takes a `sycl::queue`. SYCL itself has no
// public way to load a raw SPIR-V module -- a kernel bundle is built from
// source the runtime compiled, not from bytes handed to it -- so the image goes
// in through Level Zero, the layer SYCL sits on. It is loaded into the SYCL
// queue's own Level Zero context (read back with sycl::get_native), so the
// module, the USM allocations and oneMKL share one context and one device. A
// second context would make each side's pointers unusable by the other.
//
// Allocation policy. Every space gets a host-visible USM allocation -- one the
// device can address and the host can read -- so no program is wrong on it.
// The plan's open question about which spaces get device-only memory is
// answered where the allocation happens, and the reason the permissive answer
// is the one taken is there too.
//
// Routing. This library serves one device. A transfer is keyed on the memory
// space's canonical id: the GPU band (500) is this card, and a
// `Memory::GPU_HBM` transfer carries it, so a placed program's device memory
// lands here; every other id is memory the host owns. A dispatch runs here when
// its payload names the GPU band, or when it carries a `format=spirv` image --
// which only this backend can load, so the image itself says whose it is. A
// matmul placed on another accelerator's band is not this library's work.
// Routing every named topology is the shared work #1137 leaves open ("one
// dispatch library per build").
//
// What it refuses. A kernel whose `format=` is not `spirv`, or whose image is
// missing, is not launched -- it runs the outlined kernel on the host, the same
// road a kernel with no image takes. An `abi=` this header does not know is
// refused outright. Neither is a guess dressed up as a fallback: the format key
// exists to tell the two apart, and guessing is what it is there to prevent.
//
// Built only when the oneAPI toolchain is present; the no-SDK build is
// unchanged (build.rs).
//
//===----------------------------------------------------------------------===//

#include "vx_dispatch_plan.h"
#include "vx_host_call.h"
#include "vx_kernel_launch.h"
#include "vx_remote_routing.h"

#include <level_zero/ze_api.h>
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace {

// Diagnostics are opt-in: a test harness compares a program's stdout against
// its `// EXPECT:` lines, so an unconditional banner would fail every placed
// test. Everything here writes to stderr in any case.
bool verbose() {
  static const bool on = [] {
    const char *v = getenv("VX_DISPATCH_VERBOSE");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

#define VX_XPU_CHECK(expr)                                                     \
  do {                                                                         \
    ze_result_t vx_r_ = (expr);                                                \
    if (vx_r_ != ZE_RESULT_SUCCESS) {                                          \
      fprintf(stderr, "[Vx XPU] FATAL: %s failed with 0x%x\n", #expr,          \
              (unsigned)vx_r_);                                                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

/// The Level Zero GPU this library serves, decided once.
///
/// A build with the XPU backend compiled in can still run on a machine with no
/// Intel GPU -- the same binary, a different box -- and that is a reason to run
/// kernels on the host, not to fail. The Level Zero backend is named
/// explicitly, so a machine whose only GPU belongs to another vendor still
/// takes the host road rather than being handed a device this library cannot
/// load a SPIR-V module into.
sycl::device *xpu_device() {
  static sycl::device *device = []() -> sycl::device * {
    try {
      for (const auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
          continue;
        }
        for (const auto &dev : platform.get_devices()) {
          if (dev.is_gpu()) {
            /* Which card was picked, so a run can say what it ran on. */
            if (verbose()) {
              fprintf(stderr, "[Vx XPU] target: %s (Level Zero)\n",
                      dev.get_info<sycl::info::device::name>().c_str());
            }
            return new sycl::device(dev);
          }
        }
      }
    } catch (const sycl::exception &) {
      // No Level Zero platform at all: reported below, like no device found.
    }
    return nullptr;
  }();
  return device;
}

bool xpu_available() {
  static const bool available = [] {
    if (xpu_device()) {
      return true;
    }
    fprintf(
        stderr,
        "[Vx XPU] no Intel GPU (Level Zero); kernels will run on the host\n");
    return false;
  }();
  return available;
}

/// The one queue. Leaked on purpose: a static `sycl::queue` would be destroyed
/// at exit in an order that depends on what else is still alive, and the
/// runtime's own teardown is a worse time to find out.
sycl::queue &xpu_queue() {
  static sycl::queue *queue =
      new sycl::queue(*xpu_device(), sycl::property::queue::in_order{});
  return *queue;
}

/// Allocate on the device, reporting a refusal rather than letting a SYCL
/// exception cross the plugin ABI, which is a C boundary.
void *device_alloc(size_t bytes, bool host_visible) {
  try {
    return host_visible ? sycl::malloc_shared(bytes, xpu_queue())
                        : sycl::malloc_device(bytes, xpu_queue());
  } catch (const sycl::exception &e) {
    fprintf(stderr,
            "[Vx XPU] FATAL: could not allocate %zu bytes on the device: %s\n",
            bytes, e.what());
    abort();
  }
}

ze_context_handle_t ze_context() {
  static ze_context_handle_t context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          xpu_queue().get_context());
  return context;
}

ze_device_handle_t ze_device() {
  static ze_device_handle_t device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          xpu_queue().get_device());
  return device;
}

/// Whether a pointer is a USM allocation in this library's context, and if so
/// what kind -- device memory the host cannot read, or memory it can.
///
/// The answer decides two things: how a copy is made, and whether an outlined
/// kernel that has to run on the host can read its operands at all.
ze_memory_type_t usm_type(const void *ptr) {
  if (!ptr || !xpu_available()) {
    return ZE_MEMORY_TYPE_UNKNOWN;
  }
  ze_memory_allocation_properties_t props = {};
  props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  ze_device_handle_t owner = nullptr;
  if (zeMemGetAllocProperties(ze_context(), ptr, &props, &owner) !=
      ZE_RESULT_SUCCESS) {
    return ZE_MEMORY_TYPE_UNKNOWN;
  }
  return props.type;
}

bool is_device_ptr(const void *ptr) {
  return usm_type(ptr) == ZE_MEMORY_TYPE_DEVICE;
}

/// Whether a pointer is any USM allocation of this library's, device, shared or
/// host -- which is what decides whether a `queue::memcpy` can name it at all.
bool is_usm(const void *ptr) { return usm_type(ptr) != ZE_MEMORY_TYPE_UNKNOWN; }

/// Copy host bytes into memory on the device.
///
/// `queue::memcpy` needs USM at both ends, and a program's tensors are ordinary
/// host memory, so a device destination is reached through a USM host buffer.
/// A shared allocation is host-accessible already and needs no middle man.
void copy_into(void *dst, const void *src, size_t bytes) {
  if (!src || !dst || bytes == 0) {
    return;
  }
  if (!is_device_ptr(dst)) {
    memcpy(dst, src, bytes);
    return;
  }
  if (is_usm(src)) {
    xpu_queue().memcpy(dst, src, bytes).wait();
    return;
  }
  void *staging = sycl::malloc_host(bytes, xpu_queue());
  memcpy(staging, src, bytes);
  xpu_queue().memcpy(dst, staging, bytes).wait();
  sycl::free(staging, xpu_queue());
}

/// Copy bytes back out of memory on the device, the same way round.
void copy_out(void *dst, const void *src, size_t bytes) {
  if (!dst || !src || bytes == 0) {
    return;
  }
  if (!is_device_ptr(src)) {
    memcpy(dst, src, bytes);
    return;
  }
  if (is_usm(dst)) {
    xpu_queue().memcpy(dst, src, bytes).wait();
    return;
  }
  void *staging = sycl::malloc_host(bytes, xpu_queue());
  xpu_queue().memcpy(staging, src, bytes).wait();
  memcpy(dst, staging, bytes);
  sycl::free(staging, xpu_queue());
}

/// The Level Zero command list a launch is recorded into, and the queue it
/// executes on, created once and reused.
struct LaunchQueue {
  ze_command_queue_handle_t queue = nullptr;
  ze_command_list_handle_t list = nullptr;
};

LaunchQueue &launch_queue() {
  static LaunchQueue *lq = [] {
    auto *out = new LaunchQueue;
    ze_command_queue_desc_t qdesc = {};
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    VX_XPU_CHECK(
        zeCommandQueueCreate(ze_context(), ze_device(), &qdesc, &out->queue));
    ze_command_list_desc_t ldesc = {};
    ldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    VX_XPU_CHECK(
        zeCommandListCreate(ze_context(), ze_device(), &ldesc, &out->list));
    return out;
  }();
  return *lq;
}

/// One loaded device function, kept alive for the process because a module a
/// kernel belongs to must outlive it.
struct LoadedKernel {
  ze_module_handle_t module = nullptr;
  ze_kernel_handle_t kernel = nullptr;
};

ze_kernel_handle_t load_kernel(const char *name, const void *image,
                               size_t image_size) {
  static std::unordered_map<std::string, LoadedKernel> loaded;

  auto found = loaded.find(name);
  if (found != loaded.end()) {
    return found->second.kernel;
  }

  ze_module_desc_t mdesc = {};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  /* The image is SPIR-V, so the driver compiles it here, once per process. */
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = image_size;
  mdesc.pInputModule = static_cast<const uint8_t *>(image);

  LoadedKernel entry;
  ze_module_build_log_handle_t log = nullptr;
  ze_result_t rc =
      zeModuleCreate(ze_context(), ze_device(), &mdesc, &entry.module, &log);
  if (rc != ZE_RESULT_SUCCESS) {
    size_t log_size = 0;
    if (log) {
      zeModuleBuildLogGetString(log, &log_size, nullptr);
      std::string text(log_size + 1, '\0');
      zeModuleBuildLogGetString(log, &log_size, text.data());
      fprintf(stderr, "[Vx XPU] the driver refused the module for %s:\n%s\n",
              name, text.c_str());
    } else {
      fprintf(stderr,
              "[Vx XPU] FATAL: loading the image for %s failed with "
              "0x%x.\n",
              name, (unsigned)rc);
    }
    abort();
  }

  ze_kernel_desc_t kdesc = {};
  kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kdesc.pKernelName = name;
  rc = zeKernelCreate(entry.module, &kdesc, &entry.kernel);
  if (rc != ZE_RESULT_SUCCESS) {
    fprintf(stderr,
            "[Vx XPU] FATAL: the image loaded but has no entry named %s "
            "(0x%x).\n",
            name, (unsigned)rc);
    abort();
  }

  loaded.emplace(name, entry);
  return entry.kernel;
}

/// The byte width of each parameter the flattened argument list produced, in
/// the same order as `vx_launch_build_params` produced them.
///
/// A Level Zero launch needs each argument's size spelled out, and the size is
/// a property of the ABI tag rather than of the value: a memref contributes two
/// pointers and then 64-bit fields (offset, sizes, strides), and a scalar
/// contributes exactly its own width.
bool parameter_widths(const int32_t *arg_tags, int64_t num_args, size_t *out,
                      int *out_count) {
  int count = 0;
  for (int64_t i = 0; i < num_args; ++i) {
    const int32_t tag = arg_tags[i];
    const int width = vx_launch_param_width(tag);
    if (width == 0 || count + width > VX_LAUNCH_MAX_PARAMS) {
      return false;
    }
    if (VX_ABI_KIND(tag) == VX_ABI_KIND_MEMREF) {
      int32_t rank = VX_ABI_RANK(tag);
      out[count++] = sizeof(void *); /* allocated */
      out[count++] = sizeof(void *); /* aligned   */
      for (int32_t field = 0; field < 1 + 2 * rank; ++field) {
        out[count++] = sizeof(uint64_t);
      }
    } else {
      size_t bytes = vx_dtype_bytes(VX_ABI_KIND(tag));
      if (bytes == 0) {
        return false;
      }
      out[count++] = bytes;
    }
  }
  *out_count = count;
  return true;
}

/// Run the kernel the payload carries, on this device.
///
/// Returns false when the payload is not a SPIR-V image this library knows how
/// to launch, which leaves the caller's fallback in place rather than replacing
/// it with a guess.
bool run_device_image(const void *payload, size_t payload_size,
                      void **device_args, const int32_t *arg_tags,
                      int64_t num_args) {
  const char *kernel_name = static_cast<const char *>(payload);
  const char *format = vx_payload_field(payload, payload_size, "format=");

  /* The format key is what tells a consumer whether the bytes are something it
     can use. A format this library does not know is a refusal, not a guess at
     the loader to use. */
  if (!format || strcmp(format, "spirv") != 0) {
    if (verbose()) {
      fprintf(stderr,
              "[Vx XPU] %s has format=%s, which this backend does not load; "
              "not launching\n",
              kernel_name, format ? format : "<none>");
    }
    return false;
  }

  const void *image = nullptr;
  int64_t image_size = vx_payload_section(payload, payload_size, &image);
  if (image_size <= 0 || !image) {
    if (verbose()) {
      fprintf(stderr,
              "[Vx XPU] %s says format=spirv but carries no usable image "
              "section (code %lld); not launching\n",
              kernel_name, (long long)image_size);
    }
    return false;
  }

  /* What the image is and that it ran is a property of the kernel, not of one
     dispatch, so the opt-in diagnostic says it once rather than once per launch
     of a timing loop. */
  static std::unordered_map<std::string, bool> announced;
  const bool announce =
      verbose() && announced.emplace(kernel_name, true).second;

  if (announce) {
    fprintf(stderr, "[Vx XPU] %s: format=spirv accepted, %lld-byte image\n",
            kernel_name, (long long)image_size);
  }

  vx_launch_params params;
  if (!vx_launch_build_params(device_args, arg_tags, num_args, &params)) {
    if (verbose()) {
      fprintf(stderr,
              "[Vx XPU] %s has an argument this path cannot marshal (a "
              "publication slot); not launching\n",
              kernel_name);
    }
    return false;
  }

  /* The signature is right here in the image, so there is no reason to launch
     on faith. The flattened count is what the compiler's ABI promises and the
     image is what the driver will enter, and Level Zero cannot check one
     against the other: too few parameters and the kernel reads arguments that
     were never written. */
  const int declared =
      vx_launch_entry_param_count_spirv(image, (size_t)image_size, kernel_name);
  if (declared != params.count) {
    fprintf(stderr,
            "[Vx XPU] FATAL: %s declares %d parameters and its arguments "
            "produced %d.\n"
            "          Launching would read past what was supplied (#251).\n",
            kernel_name, declared, params.count);
    abort();
  }

  size_t widths[VX_LAUNCH_MAX_PARAMS];
  int width_count = 0;
  if (!parameter_widths(arg_tags, num_args, widths, &width_count) ||
      width_count != params.count) {
    fprintf(stderr,
            "[Vx XPU] FATAL: %s has %d parameters and their widths cover %d.\n",
            kernel_name, params.count, width_count);
    abort();
  }

  ze_kernel_handle_t kernel =
      load_kernel(kernel_name, image, (size_t)image_size);

  /* The launch geometry, from `launch=`. A strided kernel is correct under any
     configuration -- each thread walks its own stride -- so the field is a
     sizing hint with a correctness floor: an absent `launch=` means one group
     of one thread, which is what a serial region compiled against. */
  uint32_t groups = 1, group_size = 1;
  if (const char *launch = vx_payload_field(payload, payload_size, "launch=")) {
    char *rest = nullptr;
    long trip = strtol(launch, &rest, 10);
    if (rest && *rest == ',') {
      long threads = strtol(rest + 1, nullptr, 10);
      if (trip > 0 && threads > 0) {
        groups = trip > 4096 ? 4096u : (uint32_t)trip;
        group_size = threads > 1024 ? 1024u : (uint32_t)threads;
      }
    } else if (trip > 1) {
      group_size = trip < 128 ? (uint32_t)trip : 128u;
      unsigned long need = ((unsigned long)trip + group_size - 1) / group_size;
      groups = need > 4096 ? 4096u : (uint32_t)need;
    }
  }
  VX_XPU_CHECK(zeKernelSetGroupSize(kernel, group_size, 1, 1));

  for (int i = 0; i < params.count; ++i) {
    VX_XPU_CHECK(
        zeKernelSetArgumentValue(kernel, i, widths[i], params.params[i]));
  }

  LaunchQueue &lq = launch_queue();
  VX_XPU_CHECK(zeCommandListReset(lq.list));
  ze_group_count_t grid = {groups, 1, 1};
  VX_XPU_CHECK(zeCommandListAppendLaunchKernel(lq.list, kernel, &grid, nullptr,
                                               0, nullptr));
  VX_XPU_CHECK(zeCommandListClose(lq.list));
  VX_XPU_CHECK(
      zeCommandQueueExecuteCommandLists(lq.queue, 1, &lq.list, nullptr));
  VX_XPU_CHECK(zeCommandQueueSynchronize(lq.queue, UINT64_MAX));

  if (announce) {
    fprintf(stderr, "[Vx XPU] %s ran from its own image (%d params, %ux%u)\n",
            kernel_name, params.count, groups, group_size);
  }
  return true;
}

/// One row-major C[m,n] = A[m,k] * B[k,n] through oneMKL on the device.
///
/// The plan routes a classified matmul to the vendor library rather than to a
/// generated kernel, and oneMKL is the Intel parallel of cuBLAS.
template <typename T>
void gemm_on_device(int64_t m, int64_t n, int64_t k, const T *a, int64_t lda,
                    const T *b, int64_t ldb, T *c, int64_t ldc) {
  const T alpha = static_cast<T>(1);
  const T beta = static_cast<T>(0);
  auto event = oneapi::mkl::blas::row_major::gemm(
      xpu_queue(), oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans, m, n, k, alpha, a, lda, b, ldb, beta, c,
      ldc);
  event.wait();
}

/// A row-major `rows x cols` operand made available on the device.
///
/// An operand that is already USM is used where it lies; a host one is copied
/// into a packed buffer, so the device row stride is `cols` whatever the host
/// padding was. The same shape cuda_dispatch.cpp stages with, for the same
/// reason: a placed matmul's operands are host tensors unless the program
/// transferred them, and oneMKL cannot read a host pointer.
struct StagedBuffer {
  void *ptr = nullptr;
  int64_t row_stride = 0;
  bool owned = false;
  size_t bytes = 0;

  StagedBuffer() = default;
  StagedBuffer(const StagedBuffer &) = delete;
  StagedBuffer &operator=(const StagedBuffer &) = delete;

  StagedBuffer(StagedBuffer &&other) noexcept
      : ptr(other.ptr), row_stride(other.row_stride), owned(other.owned),
        bytes(other.bytes) {
    other.ptr = nullptr;
    other.owned = false;
    other.bytes = 0;
  }

  StagedBuffer &operator=(StagedBuffer &&other) noexcept {
    if (this != &other) {
      if (owned && ptr) {
        sycl::free(ptr, xpu_queue());
      }
      ptr = other.ptr;
      row_stride = other.row_stride;
      owned = other.owned;
      bytes = other.bytes;
      other.ptr = nullptr;
      other.owned = false;
      other.bytes = 0;
    }
    return *this;
  }

  ~StagedBuffer() {
    if (owned && ptr) {
      sycl::free(ptr, xpu_queue());
    }
  }
};

StagedBuffer stage(const void *host_data, int64_t rows, int64_t cols,
                   int64_t row_stride, size_t elem_bytes) {
  StagedBuffer buf;
  if (is_usm(host_data)) {
    buf.ptr = const_cast<void *>(host_data);
    buf.row_stride = row_stride;
    return buf;
  }

  size_t width = (size_t)cols * elem_bytes;
  buf.bytes = (size_t)rows * width;
  /* Host-visible rather than device-only. A program's tensor is ordinary host
     memory, and there is no USM copy that reaches it: `queue::memcpy` needs USM
     at both ends. An operand that was transferred into a device space is USM
     already and took the road above, so this is the tensor a program never
     moved -- usually the small one. */
  buf.ptr = device_alloc(buf.bytes, /*host_visible=*/true);
  if (!buf.ptr) {
    fprintf(stderr,
            "[Vx XPU] FATAL: out of device memory staging %zu bytes "
            "(%lld x %lld).\n",
            buf.bytes, (long long)rows, (long long)cols);
    abort();
  }
  buf.owned = true;
  buf.row_stride = cols;
  for (int64_t r = 0; r < rows; ++r) {
    memcpy((char *)buf.ptr + (size_t)r * width,
           (const char *)host_data + (size_t)r * row_stride * elem_bytes,
           width);
  }
  return buf;
}

/// Run a decoded plan. Returns false if it could not be run at all, in which
/// case the caller falls back to the outlined kernel.
bool run_gemm(const vx_gemm_plan &plan) {
  if (!xpu_available()) {
    return false;
  }

  size_t elem_bytes = vx_dtype_bytes(plan.dtype);
  switch (plan.dtype) {
  case VX_DTYPE_F32:
  case VX_DTYPE_F16:
    break;
  case VX_DTYPE_F64:
    /* An fp64 GEMM needs a device that has fp64 at all. The Arc does not, and
       asking oneMKL anyway kills the process with `unsupported_device`. The
       outlined kernel computes it correctly on the host instead. */
    if (!xpu_device()->has(sycl::aspect::fp64)) {
      if (verbose()) {
        fprintf(stderr, "[Vx XPU] this device has no fp64; not routing the "
                        "GEMM\n");
      }
      return false;
    }
    break;
  default:
    /* bf16 and the integer types: the entry points differ enough that
       pretending otherwise would be the guess this refusal avoids. */
    if (verbose()) {
      fprintf(stderr, "[Vx XPU] no oneMKL GEMM for %s\n",
              vx_dtype_name(plan.dtype));
    }
    return false;
  }

  if (verbose()) {
    fprintf(stderr, "[Vx XPU] GEMM %lldx%lldx%lld %s -> %s\n",
            (long long)plan.m, (long long)plan.n, (long long)plan.k,
            vx_dtype_name(plan.dtype),
            plan.out_kind == VX_GEMM_OUT_SLOT ? "slot" : "buffer");
  }

  StagedBuffer a =
      stage(plan.a_data, plan.m, plan.k, plan.a_row_stride, elem_bytes);
  StagedBuffer b =
      stage(plan.b_data, plan.k, plan.n, plan.b_row_stride, elem_bytes);

  bool out_is_slot = plan.out_kind == VX_GEMM_OUT_SLOT;
  bool out_is_usm = !out_is_slot && is_usm(plan.out_data);

  /* A USM destination is written where it lies -- `ldc` carries its row stride.
     A host tensor is not USM, so the result lands in a host-visible buffer
     first and is copied back row by row. */
  StagedBuffer c;
  if (out_is_usm) {
    c.ptr = plan.out_data;
    c.row_stride = plan.out_row_stride;
  } else {
    c.bytes = (size_t)plan.m * (size_t)plan.n * elem_bytes;
    c.ptr = device_alloc(c.bytes, /*host_visible=*/true);
    if (!c.ptr) {
      fprintf(stderr,
              "[Vx XPU] FATAL: out of device memory for a %lldx%lld result\n",
              (long long)plan.m, (long long)plan.n);
      abort();
    }
    c.owned = true;
    c.row_stride = plan.n;
  }

  void *out = c.ptr;
  try {
    switch (plan.dtype) {
    case VX_DTYPE_F32:
      gemm_on_device<float>(plan.m, plan.n, plan.k, (const float *)a.ptr,
                            a.row_stride, (const float *)b.ptr, b.row_stride,
                            static_cast<float *>(out), c.row_stride);
      break;
    case VX_DTYPE_F64:
      gemm_on_device<double>(plan.m, plan.n, plan.k, (const double *)a.ptr,
                             a.row_stride, (const double *)b.ptr, b.row_stride,
                             static_cast<double *>(out), c.row_stride);
      break;
    default:
      gemm_on_device<sycl::half>(plan.m, plan.n, plan.k,
                                 (const sycl::half *)a.ptr, a.row_stride,
                                 (const sycl::half *)b.ptr, b.row_stride,
                                 static_cast<sycl::half *>(out), c.row_stride);
      break;
    }
  } catch (const sycl::exception &e) {
    /* oneMKL refuses a call it cannot serve -- an unsupported shape, a device
       without the type -- by throwing. The outlined kernel is the answer to
       that, not a dead process, so the refusal leaves through the same door as
       every other one. */
    fprintf(stderr, "[Vx XPU] oneMKL could not run this GEMM: %s\n", e.what());
    return false;
  }

  if (out_is_usm) {
    return true;
  }

  size_t width = (size_t)plan.n * elem_bytes;
  if (out_is_slot) {
    /* Standing in for a kernel that publishes an allocation means allocating
       one and writing the descriptor. The result is copied home, because what
       the descriptor names is memory the host will read. */
    void *result = malloc((size_t)plan.m * width);
    if (!result) {
      fprintf(stderr, "[Vx XPU] FATAL: out of memory for a %lldx%lld result\n",
              (long long)plan.m, (long long)plan.n);
      abort();
    }
    memcpy(result, c.ptr, (size_t)plan.m * width);
    vx_gemm_publish_slot(&plan, result);
    return true;
  }

  /* Row by row, because the destination's row stride need not be `n`. */
  for (int64_t r = 0; r < plan.m; ++r) {
    memcpy((char *)plan.out_data + (size_t)r * plan.out_row_stride * elem_bytes,
           (const char *)c.ptr + (size_t)r * width, width);
  }
  return true;
}

/// A device operand made readable by the host for the duration of one call.
///
/// A kernel this backend cannot route still has to run, and it runs on the
/// host. Its operands may be device memory -- a `transfer` into a
/// `managed: explicit` space put them there -- which the host cannot read. So
/// each such operand is copied into a host buffer and its descriptor is pointed
/// at that buffer for the length of the call; the buffer is copied back onto
/// the device afterwards, so the program finds what the kernel wrote where it
/// expects it. Only the operand's own reachable extent is copied, computed from
/// the descriptor, so a row view costs what the row covers.
struct HostMirror {
  void **desc = nullptr;
  void *saved_allocated = nullptr;
  void *saved_aligned = nullptr;
  void *host = nullptr;
  size_t bytes = 0;
  bool active = false;
};

/// Point this operand's descriptor at host memory holding its contents.
///
/// Returns false only when the copy itself could not be made, which is out of
/// memory and fatal. An operand the host can already read is left alone.
bool mirror_operand(void *arg, int32_t tag, HostMirror *out) {
  out->active = false;
  /* A slot is where the kernel publishes a descriptor it allocated, so there
     is no existing contents to mirror. */
  if (VX_ABI_KIND(tag) != VX_ABI_KIND_MEMREF || VX_ABI_IS_SLOT(tag)) {
    return true;
  }

  const int32_t rank = (int32_t)VX_ABI_RANK(tag);
  const size_t elem_bytes = vx_dtype_bytes((int32_t)VX_ABI_ELEM(tag));
  /* The argument points at a word holding the descriptor, exactly as
     vx_launch_build_params reads it. */
  void **desc = (void **)*(void *const *)arg;
  if (!desc || elem_bytes == 0) {
    return true;
  }
  if (!is_device_ptr(vx_memref_aligned(desc))) {
    return true;
  }

  const int64_t offset = vx_memref_offset(desc);
  const int64_t *sizes = vx_memref_sizes(desc);
  const int64_t *strides = vx_memref_strides(desc, rank);
  int64_t extent = 1;
  for (int32_t d = 0; d < rank; ++d) {
    if (sizes[d] > 0) {
      extent += (sizes[d] - 1) * strides[d];
    }
  }

  size_t bytes = (size_t)(offset + extent) * elem_bytes;
  void *host = malloc(bytes);
  if (!host) {
    return false;
  }
  copy_out(host, vx_memref_aligned(desc), bytes);

  out->desc = desc;
  out->saved_allocated = desc[0];
  out->saved_aligned = desc[1];
  out->host = host;
  out->bytes = bytes;
  out->active = true;
  /* The offset stays as it is: the kernel addresses `aligned + offset`, so the
     mirror has to start where the allocation did for that arithmetic to land
     on the same elements. */
  desc[0] = host;
  desc[1] = host;
  return true;
}

void unmirror_operand(HostMirror *mirror) {
  if (!mirror->active) {
    return;
  }
  copy_into(mirror->saved_aligned, mirror->host, mirror->bytes);
  mirror->desc[0] = mirror->saved_allocated;
  mirror->desc[1] = mirror->saved_aligned;
  free(mirror->host);
  mirror->active = false;
}

} // namespace

extern "C" {

/// Allocate in this backend's memory and copy into it.
///
/// `space_access` is the model's claim about whether the host may read the
/// result, and it is what chooses the kind of allocation: a space declared
/// `managed: explicit` gets device memory the host cannot read, and every other
/// space -- `managed: cached`, or nothing said -- is host-visible. This is the
/// policy #1137 leaves to the backend, and it is decided here rather than
/// guessed from the topology, because the model is where the claim lives.
void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id,
                                   uint32_t space_access) {
  void *remote = nullptr;
  if (vx_routing_try_alloc(bytes, host_ptr, topology_id, &remote)) {
    return remote;
  }
  if (bytes == 0) {
    return nullptr;
  }

  /* Which memory this is. This library serves the GPU band: the built-in GPU is
     id 500, and a `Memory::GPU_HBM` transfer carries it, so a placed program's
     device memory lands here. Every other id -- the host, another
     accelerator's band, a NUMA node -- names memory this backend did not
     allocate and must not claim. */
  int gpu_index =
      xpu_available()
          ? vx_topology_device_index((int32_t)topology_id, VX_TOPO_GPU_BASE)
          : -1;
  if (gpu_index > 0) {
    fprintf(stderr,
            "[Vx XPU] FATAL: a transfer targets GPU %d, but this machine has "
            "one GPU.\n",
            gpu_index);
    abort();
  }
  bool on_device = gpu_index >= 0;

  if (!on_device) {
    void *ptr = malloc(bytes);
    if (ptr && host_ptr) {
      memcpy(ptr, host_ptr, bytes);
    }
    return ptr;
  }

  /* Host-visible memory, whatever the model claims, and this is where the
     plan's open question about which spaces get device memory is answered
     (§6, "The ReBAR allocation policy"). A shared USM allocation is
     addressable by the device and readable by the host, so no program is wrong
     on it. A device-only allocation is the faster choice for a large staged
     tensor and is the obvious next step, but it cannot be told apart from this
     one by topology id alone: the GPU band carries `Memory::GPU_HBM` whether
     or not the machine file declared it `managed: explicit`, and the outlined
     code for a placed tensor is free to read it from the host meanwhile --
     which is what
     tests/backend/pass/placed_tensors_are_allocated_on_their_device.vx does.
     Tightening it waits for the topology routing #1137 leaves open; until then
     the permissive allocation is the one that keeps every program correct. */
  (void)space_access;
  void *ptr = device_alloc(bytes, /*host_visible=*/true);
  if (!ptr) {
    fprintf(stderr, "[Vx XPU] FATAL: out of device memory for %zu bytes\n",
            bytes);
    abort();
  }
  copy_into(ptr, host_ptr, bytes);

  if (verbose()) {
    fprintf(stderr, "[Vx XPU] staged %zu bytes on topology %u (host-visible)\n",
            bytes, topology_id);
  }
  return ptr;
}

uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                  size_t payload_size, void **device_args,
                                  const int32_t *arg_tags, int64_t num_args) {
  const char *kernel_name = static_cast<const char *>(binary_payload);

  if (vx_routing_try_dispatch(binary_payload, payload_size, device_args,
                              arg_tags, num_args)) {
    return 1;
  }

  /* A version this header does not know describes a payload whose entries may
     not mean what this code thinks they mean, so it is refused rather than
     read. -1 is a producer that predates the key and is read as the oldest
     format, which is what it is. */
  int32_t abi = vx_payload_abi(binary_payload, payload_size);
  if (abi > VX_PAYLOAD_ABI) {
    fprintf(stderr,
            "[Vx XPU] FATAL: %s carries payload abi=%d and this runtime knows "
            "%d.\n"
            "          Refusing rather than misreading a newer format.\n",
            kernel_name, (int)abi, VX_PAYLOAD_ABI);
    abort();
  }

  /* Which device's work this is. An absent `topo=` is a producer that predates
     the key, and the GPU band is this library's own. A low band names other
     hardware -- the NPU is 100, an accelerator core 200 -- and must not be
     served by the GPU backend; a matmul placed there falls back to the outlined
     kernel, which is where it ran before this library existed. Routing every
     named topology is the shared work #1137 leaves open. */
  int32_t topo = vx_payload_topology(binary_payload, payload_size);
  bool for_this_device =
      topo < 0 || vx_topology_device_index(topo, VX_TOPO_GPU_BASE) >= 0;

  vx_gemm_plan plan;
  if (for_this_device &&
      vx_gemm_plan_decode(binary_payload, payload_size, device_args, arg_tags,
                          num_args, &plan) &&
      run_gemm(plan)) {
    return 1;
  }

  if (vx_payload_field(binary_payload, payload_size, "format=")) {
    if (run_device_image(binary_payload, payload_size, device_args, arg_tags,
                         num_args)) {
      return 1;
    }
  }

  /* Nothing here could run it, so it runs on the host. Its operands may be
     device memory -- a `transfer` into a `managed: explicit` space put them
     there -- which the host cannot read; those are copied into host buffers
     and put back after the call. Refusing instead would make a kernel that
     computes the right numbers on a machine with no backend fail on a machine
     with one, which is the opposite of what a backend is for. */
  std::vector<HostMirror> mirrors(
      (size_t)(num_args > 0 ? num_args : 1));
  for (int64_t i = 0; i < num_args; ++i) {
    if (!device_args || !device_args[i] ||
        VX_ABI_KIND(arg_tags[i]) != VX_ABI_KIND_MEMREF) {
      continue;
    }
    if (VX_ABI_IS_SLOT(arg_tags[i])) {
      /* A slot holds device memory the kernel would have published into. The
         host clone publishes host memory instead, which is only right if the
         program expects host memory there -- so a slot that holds device
         memory is a shape this fallback has no answer for, and it says so
         rather than writing a host pointer into a device space. */
      const void *slot = *(const void *const *)device_args[i];
      const void *held = slot ? vx_memref_aligned(slot) : nullptr;
      if (held &&
          is_device_ptr(vx_memref_data(held, (int32_t)VX_ABI_ELEM(arg_tags[i])))) {
        fprintf(stderr,
                "[Vx XPU] FATAL: %s was not routed, and argument %lld is a "
                "publication slot\n"
                "          holding device memory, which the host cannot "
                "address.\n",
                kernel_name, (long long)i);
        abort();
      }
      continue;
    }
    if (!mirror_operand(device_args[i], arg_tags[i], &mirrors[(size_t)i])) {
      fprintf(stderr,
              "[Vx XPU] FATAL: out of memory staging %s's argument %lld for "
              "the host\n",
              kernel_name, (long long)i);
      abort();
    }
  }

  vx_host_refuse_coop(binary_payload, payload_size, kernel_name, "XPU");
  void *kernel = vx_host_kernel_symbol(kernel_name);
  if (!kernel) {
    fprintf(stderr, "[Vx XPU] FATAL: outlined kernel %s not found\n",
            kernel_name);
    abort();
  }
  bool called = vx_host_call_kernel(
      kernel, device_args, arg_tags, num_args,
      vx_host_worker_count_for(binary_payload, payload_size));

  for (HostMirror &mirror : mirrors) {
    unmirror_operand(&mirror);
  }

  if (!called) {
    fprintf(stderr, "[Vx XPU] FATAL: could not build a call for %s\n",
            kernel_name);
    abort();
  }
  return 1;
}

uint64_t vx_plugin_dispatch_async_flat(float *xout, float *x, float *w, int n,
                                       int d) {
  // Reference matvec: xout[i] = sum_j w[i * n + j] * x[j], matching the
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
  // Dispatch synchronizes before returning, so the result is ready already.
  (void)future_id;
}

int32_t vx_plugin_transfer_device_to_host(void *device_ptr, void *host_ptr,
                                          size_t bytes, uint32_t topology_id) {
  if (vx_routing_try_fetch(device_ptr, host_ptr, bytes, topology_id)) {
    return 1;
  }
  vx_routing_refuse_handle("a read-back", device_ptr, topology_id);
  copy_out(host_ptr, device_ptr, bytes);
  return 1;
}

void *vx_plugin_transfer_peer(void *src_device_ptr, uint32_t src_topology_id,
                              uint32_t dst_topology_id, size_t bytes) {
  void *routed = nullptr;
  if (vx_routing_try_peer(src_device_ptr, src_topology_id, dst_topology_id,
                          bytes, &routed)) {
    return routed;
  }
  vx_routing_refuse_handle("a peer handoff", src_device_ptr, src_topology_id);
  /* One device, so both topologies name it and the movement between them is a
     copy. Answering rather than refusing keeps a program written for two
     devices correct on a machine that has one. */
  void *dst = malloc(bytes);
  if (dst && src_device_ptr) {
    copy_out(dst, src_device_ptr, bytes);
  }
  return dst;
}

void vx_plugin_free(void *device_ptr, uint32_t topology_id) {
  if (!device_ptr) {
    return;
  }
  if (vx_routing_try_free(device_ptr, topology_id)) {
    return;
  }
  vx_routing_refuse_handle("a free", device_ptr, topology_id);
  /* Platform memory an ordinary `malloc` can release, and USM the runtime must.
     Which one this is is a property of the pointer, not of the topology: a
     pointer is freed by the allocator that made it. */
  if (usm_type(device_ptr) != ZE_MEMORY_TYPE_UNKNOWN) {
    sycl::free(device_ptr, xpu_queue());
  } else {
    free(device_ptr);
  }
}

void vx_plugin_release_future(uint64_t future_id) { (void)future_id; }

int32_t vx_plugin_control(uint32_t opcode, void *payload) {
  (void)payload;
  switch (opcode) {
  case VX_CTRL_GET_DEVICE_COUNT:
    return xpu_available() ? 1 : 0;
  case VX_CTRL_INIT_DEVICE:
    return xpu_available() ? 1 : 0;
  case VX_CTRL_SHUTDOWN:
    return 1;
  default:
    return 0;
  }
}

} // extern "C"
