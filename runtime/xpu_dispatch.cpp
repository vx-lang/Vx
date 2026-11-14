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
// payload. This library takes the image out of the payload's section, loads it,
// launches it with the flattened argument list, and runs a `kind=matmul` on
// oneMKL.
//
// Two runtimes, for one reason: oneMKL takes a `sycl::queue`, and SYCL has no
// public way to load a raw SPIR-V module -- a kernel bundle is built from
// source it compiled, not from bytes handed to it. So the image goes in through
// Level Zero, the layer SYCL sits on, into the queue's own Level Zero context
// (read back with sycl::get_native). The module, the USM allocations and oneMKL
// then share one context and one device. A second context would make each
// side's pointers unusable by the other.
//
// Allocation. Every space this library serves gets shared USM, which the device
// can address and the host can read, so no program is wrong on it. Device-only
// memory is faster for a large staged tensor and is not used yet -- the reason
// is at the allocation. A declared space the program did not alias to this
// device is refused when the program declared this device one of its own and
// the space says the host cannot read it, because host memory there is memory a
// kernel would read as the device's.
//
// Routing. This library serves one device, the Intel GPU, whose dispatch id is
// the SPIR-V band's `VX_TOPO_SPIRV_BASE` (1000). That id is not the CUDA
// backend's: 500 is the NVIDIA GPU, and the CUDA library claims every pair it
// is given, so a SPIR-V topology aliased to 500 had its tensors allocated with
// `cudaMalloc` while its kernels ran here. Every id outside 1000..1099 is
// memory the host owns. A dispatch runs here when its payload names the SPIR-V
// band, or when it carries a `format=spirv` image, which only this backend can
// load.
//
// A topology a machine file declares has an id that is a hash of its name, so
// the program sends the id it stands for at the start of `main`
// (`VX_CTRL_LOCAL_DEVICE`). Without that alias a program naming this GPU
// through a machine file would have its device tensors allocated on the host.
//
// What it refuses. A kernel whose `format=` is not `spirv`, or whose image is
// missing, is not launched here: it runs its host copy, which is the road a
// kernel with no image takes anyway. A payload `abi=` this build does not know
// is refused instead, because a newer format may mean something else by the
// entries it read.
//
// The rest of the ABI. A collective is a copy and a job is one rank (there is
// one GPU here and no library that joins several), and the count of a kernel
// with code after its blocks is kept between launches -- both as on a CPU
// backend, so a program that uses them links and runs the same way here.
//
// Built only when the oneAPI toolchain is present; the no-SDK build is
// unchanged (build.rs).
//
//===----------------------------------------------------------------------===//

#include "vx_arrival_counts.h"
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
#include <mutex>
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

/// Allocate memory the device can address and the host can read.
///
/// Shared USM is the only kind this library allocates, so every pointer it
/// hands a program is memory both sides can use. Device-only memory is the
/// faster choice for a large staged tensor and is not used yet: the reason is
/// at `vx_plugin_alloc_and_transfer`.
///
/// The refusal is reported here because a SYCL exception must not cross the
/// plugin ABI, which is a C boundary.
void *device_alloc(size_t bytes) {
  try {
    return sycl::malloc_shared(bytes, xpu_queue());
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
/// The kind decides whether an outlined kernel that has to run on the host can
/// read its operands. Every allocation this library makes is memory it can, so
/// the device-memory answer is only ever a check that fails loudly.
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
/// host -- which is what decides whether the device can address it.
bool is_usm(const void *ptr) { return usm_type(ptr) != ZE_MEMORY_TYPE_UNKNOWN; }

/// Runs before the one-rank collective of `vx_collective_cpu.h` copies its
/// input to its output. A launch here waits before it returns, so the tensors
/// the GPU wrote are finished, and this normally has nothing to wait for.
///
/// The copy is made by the host. Every allocation in the SPIR-V band is shared
/// USM, which the host may read, so a collective can always make it. The check
/// below is therefore a guard and not a road a run takes today: it is here
/// because device-only allocation would make the copy impossible, and reading
/// memory the host cannot address would otherwise be the way that is found out.
void xpu_collective_before_copy(const void *input, const void *output,
                                size_t bytes) {
  (void)bytes;
  if (!xpu_available()) {
    return;
  }
  try {
    xpu_queue().wait();
  } catch (const sycl::exception &e) {
    fprintf(stderr,
            "[Vx XPU] FATAL: a collective could not wait for the device: %s\n",
            e.what());
    abort();
  }
  if (is_device_ptr(input) || is_device_ptr(output)) {
    fprintf(stderr,
            "[Vx XPU] FATAL: a collective got device memory the host cannot "
            "read,\n          and a job of one rank is a copy the host makes. "
            "A collective needs shared USM.\n");
    abort();
  }
}

/// Copy bytes between host memory and this device's memory, either direction.
///
/// Every allocation this library makes is shared USM, which the host can read
/// and write, so a copy in either direction is a `memcpy`. This used to have a
/// second path for device-only memory: a `malloc_host` staging buffer with a
/// `queue::memcpy` on each side. Nothing allocates device-only memory, so no
/// run reached it, and the path also dereferenced the staging pointer without
/// checking it, which an out-of-memory machine would have found first.
void copy_bytes(void *dst, const void *src, size_t bytes) {
  if (!dst || !src || bytes == 0) {
    return;
  }
  memcpy(dst, src, bytes);
}

/// The Level Zero command list a launch is recorded into, and the queue it
/// executes on, created once and reused.
struct LaunchQueue {
  ze_command_queue_handle_t queue = nullptr;
  ze_command_list_handle_t list = nullptr;
};

/// Declared topology ids the program said are this machine's GPU, each with the
/// built-in id it stands for (`VX_CTRL_LOCAL_DEVICE`).
///
/// A declared id is a hash of a name, so it is not in a device band and nothing
/// about it says "GPU". A program that names the Arc through a machine file
/// sends the pair at the start of `main`, and without it every transfer to that
/// topology would land in host memory -- memory the device cannot be launched
/// at.
struct LocalDevice {
  int32_t declared;
  int32_t builtin;
};

constexpr size_t kMaxLocalDevices = 64;
LocalDevice local_devices[kMaxLocalDevices];
size_t local_device_count = 0;
std::mutex local_devices_mutex;

/// The built-in topology id `topology_id` stands for: the id a declared
/// topology was aliased to, or the id itself.
int32_t builtin_topology(int32_t topology_id) {
  std::lock_guard<std::mutex> lock(local_devices_mutex);
  for (size_t i = 0; i < local_device_count; ++i) {
    if (local_devices[i].declared == topology_id) {
      return local_devices[i].builtin;
    }
  }
  return topology_id;
}

/// Whether the program said that one of its topologies, or one of its memory
/// spaces, is the device this library serves.
///
/// This separates a program that means a device from one that only wrote a
/// `Memory` block: a declared topology with no `arch:` sends no pair, its
/// kernels run on the host, and its memories are a model of a machine rather
/// than this card's (`a_shared_edge_lets_the_cpu_read_a_borrowed_gpu_tensor` in
/// tests/optimizations is that shape).
///
/// Which device a pair stands for is the second number: 1000 for a `spirv64`
/// topology and 500 for an `nvptx64` one, and a program sends every pair to
/// whichever library it runs with. Only a pair in the SPIR-V band is this card.
/// A program that declared an `nvptx64` topology declared another backend's
/// device, so its spaces are host memory here, which is what the CPU backends
/// answer with.
bool declared_this_device() {
  std::lock_guard<std::mutex> lock(local_devices_mutex);
  for (size_t i = 0; i < local_device_count; ++i) {
    if (vx_topology_device_index(local_devices[i].builtin,
                                 VX_TOPO_SPIRV_BASE) >= 0) {
      return true;
    }
  }
  return false;
}

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
  /// How many parameters the image declares.
  ///
  /// Reading it means walking the whole SPIR-V module, and it is a property of
  /// the kernel, so it is read once here and kept. The cache key holds the
  /// kernel's name and the library its payload came from, so two launches that
  /// find the same entry have the same image and the same count.
  int param_count = -1;
};

LoadedKernel load_kernel(const std::string &key, const char *name,
                         const void *image, size_t image_size) {
  /* The caller holds the launch lock for the whole call, which is what keeps
     this map safe. */
  static std::unordered_map<std::string, LoadedKernel> loaded;

  auto found = loaded.find(key);
  if (found != loaded.end()) {
    return found->second;
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

  /* A build log is a live handle even when the build succeeded -- it is where
     the driver's warnings are -- and nothing here reads it after this point. */
  if (log) {
    zeModuleBuildLogDestroy(log);
  }

  entry.param_count =
      vx_launch_entry_param_count_spirv(image, image_size, name);

  loaded.emplace(key, entry);
  return entry;
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
/// to launch, or when this machine has no device to launch it on, which leaves
/// the caller's host copy to do the work.
bool run_device_image(const void *payload, size_t payload_size,
                      void **device_args, const int32_t *arg_tags,
                      int64_t num_args) {
  /* A build with this backend in it still runs on a machine with no Intel GPU:
     the same binary, a different box, and that is a reason to run the kernel on
     the host rather than to fail. Everything below reaches `ze_context()`
     through `xpu_queue()`, which dereferences the device pointer `xpu_device()`
     returns, so this check has to come before any of them -- without it a
     machine with oneAPI and no Intel GPU crashes on its first placed kernel. */
  if (!xpu_available()) {
    return false;
  }

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

  /* The key of everything this launch remembers about a kernel: its name and
     the library the payload came from. The library is in the key because two
     Vx libraries can each have a `vx_npu_kernel_0`, and a name-only key runs
     the first library's kernel for the second library's call -- the parameter
     check then reads the second image, so matching signatures hide the swap.
     The CUDA cache keys on the same three things; the device is not one of them
     here because `xpu_device()` has already picked the only one this library
     serves. */
  char library[32];
  snprintf(library, sizeof library, "%p", vx_payload_library(payload));
  const std::string key = std::string(kernel_name) + "@" + library;

  /* One lock for the whole launch, the way cuda_dispatch.cpp locks its kernel
     cache. Tasks can dispatch from several threads at once and everything below
     is shared between them: both maps, the kernel handle that
     `zeKernelSetArgumentValue` writes, and the one command list the launch is
     recorded into. Level Zero does not allow two threads in one handle or one
     list at a time -- a second thread's argument would land between the first
     thread's arguments and its launch, and its `zeCommandListReset` would erase
     a list that was half appended. Each launch here waits before it returns, so
     one lock held from the load to the wait is enough. */
  static std::mutex launch_mutex;
  std::lock_guard<std::mutex> lock(launch_mutex);

  /* What the image is and that it ran is a property of the kernel, not of one
     dispatch, so the opt-in diagnostic says it once rather than once per launch
     of a timing loop. */
  static std::unordered_map<std::string, bool> announced;
  const bool announce = verbose() && announced.emplace(key, true).second;

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

  size_t widths[VX_LAUNCH_MAX_PARAMS];
  int width_count = 0;
  if (!parameter_widths(arg_tags, num_args, widths, &width_count) ||
      width_count != params.count) {
    fprintf(stderr,
            "[Vx XPU] FATAL: %s has %d parameters and their widths cover %d.\n",
            kernel_name, params.count, width_count);
    abort();
  }

  /* Every pointer the kernel dereferences has to be memory the device can
     address. The driver does not refuse a host address: the kernel reads memory
     the GPU does not know, the process dies of a page fault, and nothing in the
     message says which launch did it. Every allocation this library makes in
     the SPIR-V band is USM, so a non-USM pointer here is memory of the
     program's own, and the kernel that reads it is one the host has to run. */
  for (int64_t i = 0; i < num_args; ++i) {
    if (VX_ABI_KIND(arg_tags[i]) != VX_ABI_KIND_MEMREF) {
      continue;
    }
    const void *desc = *(const void *const *)device_args[i];
    if (!is_usm(vx_memref_aligned(desc))) {
      if (verbose()) {
        fprintf(stderr,
                "[Vx XPU] %s: argument %lld is at an address the device "
                "cannot reach; not launching\n",
                kernel_name, (long long)i);
      }
      return false;
    }
  }

  /* The signature is right here in the image, so there is no reason to launch
     on faith. The flattened count is what the compiler's ABI promises and the
     image is what the driver will enter, and Level Zero cannot check one
     against the other: too few parameters and the kernel reads arguments that
     were never written. The count is a property of the kernel, so the load
     reads it once and this reads it from the cache entry: doing the scan here
     instead walked the whole image on every launch, inside the lock a launch
     holds. */
  LoadedKernel loaded =
      load_kernel(key, kernel_name, image, (size_t)image_size);
  ze_kernel_handle_t kernel = loaded.kernel;
  if (loaded.param_count != params.count) {
    fprintf(stderr,
            "[Vx XPU] FATAL: %s declares %d parameters and its arguments "
            "produced %d.\n"
            "          Launching would read past what was supplied.\n",
            kernel_name, loaded.param_count, params.count);
    abort();
  }

  /* The launch geometry, all three axes. `grid=` and `block=` are the shape a
     placed region's `across` loop asked for, so they are read first: they are
     the sizes the program gave. `launch=` is something else -- the trip count
     of one loop the frontend proved stridable, a sizing hint with a correctness
     floor -- and a region with blocks that do not stride would compute part of
     its answer, or none, at a shape it did not ask for. A missing `grid=` means
     one group of one thread, which is what a serial region compiled against.

     All three axes go to the launch because the payload can name all three.
     This used to keep only the first, so `grid=8,4,1` ran eight groups instead
     of thirty-two and left part of the output unwritten without a word.

     No `spirv64` program writes `grid=` today: only an `across` loop does, and
     the compiler refuses that loop on this arch
     (tests/frontend/fail/an_across_loop_on_a_spirv64_topology_is_refused.vx).
     A hand-written payload is the only way to reach a three-axis grid, and the
     launch prints the axes it used, so a run that gets one says so. */
  unsigned grid[3] = {1, 1, 1};
  unsigned group_size = 1;
  char why[160];
  const int set_by_program =
      vx_launch_shape(payload, payload_size, device_args, arg_tags, num_args,
                      grid, &group_size, why, sizeof why);
  if (set_by_program < 0) {
    fprintf(stderr, "[Vx XPU] FATAL: %s: %s\n", kernel_name, why);
    abort();
  }
  if (!set_by_program) {
    if (const char *launch =
            vx_payload_field(payload, payload_size, "launch=")) {
      char *rest = nullptr;
      long trip = strtol(launch, &rest, 10);
      if (rest && *rest == ',') {
        long threads = strtol(rest + 1, nullptr, 10);
        if (trip > 0 && threads > 0) {
          grid[0] = trip > 4096 ? 4096u : (uint32_t)trip;
          group_size = threads > 1024 ? 1024u : (uint32_t)threads;
        }
      } else if (trip > 1) {
        group_size = trip < 128 ? (uint32_t)trip : 128u;
        unsigned long need =
            ((unsigned long)trip + group_size - 1) / group_size;
        grid[0] = need > 4096 ? 4096u : (uint32_t)need;
      }
    }
  }
  VX_XPU_CHECK(zeKernelSetGroupSize(kernel, group_size, 1, 1));

  for (int i = 0; i < params.count; ++i) {
    VX_XPU_CHECK(
        zeKernelSetArgumentValue(kernel, i, widths[i], params.params[i]));
  }

  LaunchQueue &lq = launch_queue();
  VX_XPU_CHECK(zeCommandListReset(lq.list));
  ze_group_count_t group_count = {grid[0], grid[1], grid[2]};
  VX_XPU_CHECK(zeCommandListAppendLaunchKernel(lq.list, kernel, &group_count,
                                               nullptr, 0, nullptr));
  VX_XPU_CHECK(zeCommandListClose(lq.list));
  VX_XPU_CHECK(
      zeCommandQueueExecuteCommandLists(lq.queue, 1, &lq.list, nullptr));
  VX_XPU_CHECK(zeCommandQueueSynchronize(lq.queue, UINT64_MAX));

  if (announce) {
    fprintf(stderr,
            "[Vx XPU] %s ran from its own image (%d params, %ux%ux%u groups "
            "of %u)\n",
            kernel_name, params.count, grid[0], grid[1], grid[2], group_size);
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
  /* A program's tensor is ordinary host memory, and there is no USM copy that
     reaches it: `queue::memcpy` needs USM at both ends. An operand that was
     transferred into a device space is USM already and took the road above, so
     this is the tensor a program never moved -- usually the small one. */
  buf.ptr = device_alloc(buf.bytes);
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
    break;
  case VX_DTYPE_F16:
    /* A half-precision GEMM needs fp16 arithmetic on the device. On a device
       without it oneMKL refuses the call, and the outlined kernel computes the
       same answer on the host, so this does not make the call at all. */
    if (!xpu_device()->has(sycl::aspect::fp16)) {
      if (verbose()) {
        fprintf(stderr, "[Vx XPU] this device has no fp16; not routing the "
                        "GEMM\n");
      }
      return false;
    }
    break;
  case VX_DTYPE_F64:
    /* An fp64 GEMM needs a device that has fp64 at all. The Arc does not, so
       the call is not made: the outlined kernel computes it correctly on the
       host instead. */
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

  /* Where the GEMM writes.
     A USM destination is written where it lies -- `ldc` carries its row stride.
     A host tensor is not USM, so the result lands in a host-visible buffer
     first and is copied back row by row. A slot is a published allocation, so
     the buffer the GEMM writes is the one the program receives: this used to
     allocate `c`, run the GEMM into it, allocate a second buffer of the same
     size and copy `c` across, which doubled the peak and copied every result
     byte once more for nothing. It is shared USM rather than `malloc` because
     the descriptor is memory a later placed kernel may read: a host pointer in
     it is an address the GPU cannot reach. */
  StagedBuffer c;
  if (out_is_usm) {
    c.ptr = plan.out_data;
    c.row_stride = plan.out_row_stride;
  } else {
    c.bytes = (size_t)plan.m * (size_t)plan.n * elem_bytes;
    c.ptr = device_alloc(c.bytes);
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
       that, not a dead process, so this returns false like the other refusals
       and the caller runs the host copy. */
    fprintf(stderr, "[Vx XPU] oneMKL could not run this GEMM: %s\n", e.what());
    return false;
  } catch (const std::exception &e) {
    /* The other refusal oneMKL makes is `oneapi::mkl::exception`, which
       derives from `std::exception` and not from SYCL's. It used to escape
       `vx_plugin_dispatch_async`, which is a C function, and an exception that
       crosses one calls `std::terminate`: the program died instead of running
       the host copy. Anything standard that reaches here is a refusal to run
       this GEMM, so it takes the same answer. */
    fprintf(stderr, "[Vx XPU] oneMKL could not run this GEMM: %s\n", e.what());
    return false;
  }

  if (out_is_usm) {
    return true;
  }

  if (out_is_slot) {
    /* The buffer the GEMM wrote is the allocation the program receives, so its
       ownership moves to the program with the descriptor. */
    c.owned = false;
    vx_gemm_publish_slot(&plan, c.ptr);
    return true;
  }

  size_t width = (size_t)plan.n * elem_bytes;
  /* Row by row, because the destination's row stride need not be `n`. */
  for (int64_t r = 0; r < plan.m; ++r) {
    memcpy((char *)plan.out_data + (size_t)r * plan.out_row_stride * elem_bytes,
           (const char *)c.ptr + (size_t)r * width, width);
  }
  return true;
}

/// Refuse a host run whose operands the host cannot read.
///
/// A kernel this backend cannot route still has to run, and it runs on the
/// host. Its operands are shared USM, which the host can read, so this normally
/// finds nothing. It is here because device-only allocation would make a host
/// copy of such a kernel impossible, and a kernel reading memory it cannot
/// address is not a failure anyone can trace back from.
///
/// Copying each operand to host memory and putting it back afterwards was
/// written first, and there is no run that reaches it: every allocation in the
/// SPIR-V band is shared USM. It was removed rather than left standing, because
/// code no test can reach is code no test keeps right. When device-only
/// allocation lands, the mirroring is what has to come back here.
void refuse_device_operands_on_host(void *const *device_args,
                                    const int32_t *arg_tags, int64_t num_args,
                                    const char *kernel_name) {
  for (int64_t i = 0; i < num_args; ++i) {
    const int32_t tag = arg_tags[i];
    if (VX_ABI_KIND(tag) != VX_ABI_KIND_MEMREF) {
      continue;
    }
    /* A slot holds either nothing yet or the descriptor the kernel published.
       What it holds is device memory only if the program put it there, and
       that is the case the check below is for. */
    const void *slot = *(const void *const *)device_args[i];
    const void *desc = VX_ABI_IS_SLOT(tag) ? vx_memref_aligned(slot) : slot;
    if (!desc) {
      continue;
    }
    if (is_device_ptr(vx_memref_aligned(desc))) {
      fprintf(stderr,
              "[Vx XPU] FATAL: %s was not routed, and argument %lld is "
              "device-only memory\n"
              "          the host cannot address. A host copy needs shared "
              "USM.\n",
              kernel_name, (long long)i);
      abort();
    }
  }
}

} // namespace

extern "C" {

/// Allocate in this backend's memory and copy into it.
///
/// `space_access` is the model's claim about whether the host may read the
/// result. In the SPIR-V band it is ignored: every allocation there is shared
/// USM, which the device can address and the host can read, so no program is
/// wrong on it. Outside the band it decides whether host memory can answer a
/// request at all -- see the refusal below.
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

  /* Which memory this is. This library serves the SPIR-V band: 1000 is the
     Intel GPU's own id, and a declared topology's id is a hash of its name, so
     the program aliased the hash to 1000 at startup and the alias is applied
     first -- otherwise a program that names this GPU through a machine file
     would have its device tensors allocated on the host. Every other id -- the
     host, another accelerator's band, the CUDA backend's 500, a NUMA node --
     names memory this backend did not allocate and must not claim. */
  const int32_t requested = (int32_t)topology_id;
  const int32_t builtin = builtin_topology(requested);
  int gpu_index = xpu_available()
                      ? vx_topology_device_index(builtin, VX_TOPO_SPIRV_BASE)
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
    /* A declared space the program did not alias. Its id is a hash, so this
       library cannot tell whether it names the device or the host, and that is
       the one fact host memory turns on: the program declared it `managed:
       explicit`, which says a kernel reads it as device memory. Handing back a
       `malloc` buffer makes the region run on the host or read bytes the device
       never wrote, and before the launch check existed it made the device read
       zeros. Refuse by name instead of guessing, because the table that would
       map the hash lives in the compiler.

       Only when the program said this device is one of its own. A program that
       declared none is modelling a machine with no accelerator on this side:
       its kernels run on the host and its memories are a description, so host
       memory answers it correctly. A program written for another backend's
       device has the same shape, and it does send pairs: `nvptx64` sends 500,
       the NVIDIA GPU. So this asks whether a pair names the SPIR-V band, not
       whether the program sent any pair at all.

       CUDA reports the mirror case -- device memory where the model promised
       the host could read -- the same way: the topology, the bytes, what the
       program declared, and what to change. Here the answer would be host
       memory the program cannot use, so it is fatal rather than a warning. */
    const bool declared = requested >= VX_TOPO_NAME_BASE;
    if (xpu_available() && declared && builtin == requested &&
        declared_this_device() && space_access == VX_SPACE_ACCESS_EXPLICIT) {
      fprintf(
          stderr,
          "[Vx XPU] FATAL: a transfer of %zu bytes asks for memory space "
          "%d (its topology), and this backend cannot place that space on "
          "the GPU.\n"
          "          The id is a hash of a declared space's name, and the "
          "program did not tell this library which device it stands for: "
          "no VX_CTRL_LOCAL_DEVICE pair named it.\n"
          "          The space is declared `managed: explicit`, so a kernel "
          "reads it as device memory. Host memory here would run the "
          "region on the host, or let the device read bytes nothing "
          "wrote.\n"
          "          Make the space the topology's `memory:`, so the "
          "compiler sends the pair.\n",
          bytes, requested);
      abort();
    }
    void *ptr = malloc(bytes);
    if (ptr && host_ptr) {
      memcpy(ptr, host_ptr, bytes);
    }
    return ptr;
  }

  /* Shared USM, whatever the model claims. The device can address it and the
     host can read it, so no program is wrong on it. Device-only memory is the
     faster choice for a large staged tensor and is not used yet: an id in this
     band says which device, not which kind of memory the space declared, so
     `managed: explicit` and the lack of it arrive the same way, and the host
     copy of a placed tensor reads this memory meanwhile -- which is what
     tests/backend/pass/placed_tensors_are_allocated_on_their_device.vx does.
     Until the space itself says which kind it wants, the permissive allocation
     is the one that keeps every program correct. */
  (void)space_access;
  void *ptr = device_alloc(bytes);
  if (!ptr) {
    fprintf(stderr, "[Vx XPU] FATAL: out of device memory for %zu bytes\n",
            bytes);
    abort();
  }
  copy_bytes(ptr, host_ptr, bytes);

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
     the key, and the SPIR-V band is this library's own. A low band names other
     hardware -- the NPU is 100, an accelerator core 200, the CUDA backend's GPU
     500 -- and must not be served by this backend; a matmul placed there falls
     back to the outlined kernel, which is where it ran before this library
     existed. Serving every named topology needs one dispatch library per build,
     which is not what this is. */
  int32_t topo = vx_payload_topology(binary_payload, payload_size);
  bool for_this_device =
      topo < 0 ||
      vx_topology_device_index(builtin_topology(topo), VX_TOPO_SPIRV_BASE) >= 0;

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

  /* Nothing here could run it, so it runs on the host. Its operands are shared
     USM, which the host can read, so this normally checks nothing at all. A
     host-only pointer would mean the device could not have run the kernel
     either, and the check says so rather than letting the call read a pointer
     it cannot follow. */
  if (device_args) {
    refuse_device_operands_on_host(device_args, arg_tags, num_args,
                                   kernel_name);
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
      vx_host_worker_count_for(binary_payload, payload_size, device_args,
                               arg_tags, num_args),
      vx_host_flushes_subnormals(binary_payload, payload_size));

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
  // signature the flat path emits.
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
  copy_bytes(host_ptr, device_ptr, bytes);
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
  /* This machine has one GPU, so a handoff into the SPIR-V band lands on the
     same card and is a copy. The destination has to be memory the device can
     address, because the descriptor that names it can reach a later placed
     kernel: a `malloc` buffer there is an address the GPU cannot read, and the
     fault says nothing about which launch took it.
     `vx_plugin_alloc_and_transfer` refuses a GPU index above 0, so the band
     this can name is the first one. */
  const bool to_device =
      xpu_available() &&
      vx_topology_device_index(builtin_topology((int32_t)dst_topology_id),
                               VX_TOPO_SPIRV_BASE) >= 0;
  void *dst = to_device ? device_alloc(bytes) : malloc(bytes);
  if (dst && src_device_ptr) {
    copy_bytes(dst, src_device_ptr, bytes);
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

/// No tensor maps without a CUDA GPU: only a kernel compiled for `sm_90a` or a
/// later chip is handed one, and this library loads SPIR-V images, so it never
/// receives such a kernel. A kernel it cannot load runs on the host, and the
/// host copy of the kernel never reads a map.
int64_t vx_plugin_tensor_map(int64_t base, int64_t rank, int64_t elem_bytes,
                             int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                             int64_t d4, int64_t s0, int64_t s1, int64_t s2,
                             int64_t s3, int64_t s4, int64_t b0, int64_t b1,
                             int64_t b2, int64_t b3, int64_t b4) {
  (void)base, (void)rank, (void)elem_bytes, (void)d0, (void)d1, (void)d2;
  (void)d3, (void)d4, (void)s0, (void)s1, (void)s2, (void)s3, (void)s4;
  (void)b0, (void)b1, (void)b2, (void)b3, (void)b4;
  return 0;
}

/// `vx_cuda_set_stream` selects a CUDA stream for the calls of the calling
/// thread. This library has no CUDA stream: its launches go on one queue and
/// wait before they return, so there is nothing to select and null is the
/// answer that says so. The pair is here so that a host program links on every
/// build, for the same reason the CPU backends carry it.
void vx_cuda_set_stream(void *stream) { (void)stream; }
void *vx_cuda_get_stream(void) { return nullptr; }

/// The counts that no launch is using, one for each `spawn` site. A kept count
/// is 0 -- the last pass of a launch sets it back -- so a kernel with code
/// after its blocks allocates once rather than once per launch.
static vx::ArrivalCounts &arrival_counts() {
  static vx::ArrivalCounts counts;
  return counts;
}

void *vx_plugin_arrivals_take(const void *site, size_t bytes, void *zeros,
                              uint32_t topology_id, uint32_t space_access) {
  /* The launch that uses this count goes on the one queue, and a launch waits
     before it returns, so a count given back is free by the time the next
     launch of the same site takes it. */
  if (void *kept = arrival_counts().take(site, topology_id, nullptr, bytes)) {
    return kept;
  }
  return vx_plugin_alloc_and_transfer(bytes, zeros, topology_id, space_access);
}

void vx_plugin_arrivals_give(const void *site, void *count, size_t bytes,
                             uint32_t topology_id) {
  // Keep only a count in this process. A count on a worker of the fleet is the
  // worker's memory, which this process can neither keep nor free.
  void *spare =
      vx_routing_worker((int32_t)topology_id)
          ? count
          : arrival_counts().give(site, topology_id, nullptr, count, bytes);
  if (spare) {
    vx_plugin_free(spare, topology_id);
  }
}

int32_t vx_plugin_control(uint32_t opcode, void *payload) {
  switch (opcode) {
  case VX_CTRL_GET_DEVICE_COUNT:
    return xpu_available() ? 1 : 0;
  case VX_CTRL_INIT_DEVICE:
    return xpu_available() ? 1 : 0;
  case VX_CTRL_LOCAL_DEVICE: {
    /* The pair a program sends for a topology its machine file declares: the
       declared id, then the built-in id it stands for. Kept, so that a
       transfer to the declared id allocates the GPU's memory. A program sends
       the same pairs again from each `pub fn`, so an id already known is a
       success rather than a second entry. */
    const int32_t *pair = static_cast<const int32_t *>(payload);
    if (!pair) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(local_devices_mutex);
    for (size_t i = 0; i < local_device_count; ++i) {
      if (local_devices[i].declared == pair[0]) {
        return 1;
      }
    }
    if (local_device_count == kMaxLocalDevices) {
      return 0;
    }
    local_devices[local_device_count++] = {pair[0], pair[1]};
    return 1;
  }
  case VX_CTRL_SHUTDOWN:
    return 1;
  default:
    return 0;
  }
}

} // extern "C"

/* The collectives of docs/lang/abi.md section 3.7. This library serves one GPU
   and links no library that joins several ranks, so a job here is one rank and
   each collective is a copy -- the answer a CPU backend gives, and the one the
   Apple backend gives. A program that asks for more ranks is refused there
   rather than being run with the wrong arithmetic. */
#define VX_COLLECTIVE_CPU_BEFORE_COPY(input, output, bytes)                    \
  xpu_collective_before_copy((input), (output), (bytes))
#include "vx_collective_cpu.h"
