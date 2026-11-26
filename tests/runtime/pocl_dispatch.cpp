//===- pocl_dispatch.cpp - Run SPIR-V images on pocl ------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A dispatch library for tests on a machine with no Intel GPU. It is the x86
// host library, except that a kernel with a SPIR-V image runs that image on
// pocl, an OpenCL implementation that runs kernels on the CPU. So a test runs
// the same image that runtime/xpu_dispatch.cpp gives to the Arc.
//
// The launch shape is the one that runtime/xpu_dispatch.cpp computes. Keep the
// two the same. pocl's CPU device can address the memory of the program, so
// each pointer of a tensor goes to the kernel as it is.
//
// At exit, the library prints how many kernels ran from their image, and how
// many SPIR-V kernels ran on the host copy instead.
// tests/integration_test/spirv_images_on_pocl.rs reads that line.
//
//===----------------------------------------------------------------------===//

#define VX_BACKEND_NAME "x86"
#define VX_BACKEND_ALIGN 64

#define vx_plugin_dispatch_async vx_host_dispatch_async
#include "host_dispatch_common.h"
#undef vx_plugin_dispatch_async

#include "vx_kernel_launch.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::atomic<int> g_ran{0}, g_on_host{0};

[[noreturn]] void fatal(const char *what, const char *name, int err) {
  fprintf(stderr, "[pocl] FATAL: %s for %s: %d\n", what, name, err);
  abort();
}

struct Cl {
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  clSetKernelArgDevicePointerEXT_fn set_device_pointer = nullptr;
};

Cl &cl() {
  static Cl c = [] {
    Cl c;
    cl_uint count = 0;
    clGetPlatformIDs(0, nullptr, &count);
    std::vector<cl_platform_id> platforms(count);
    clGetPlatformIDs(count, platforms.data(), nullptr);
    cl_platform_id pocl = nullptr;
    for (cl_platform_id p : platforms) {
      char name[256] = {0};
      clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof name - 1, name, nullptr);
      if (strstr(name, "Portable Computing Language")) {
        pocl = p;
      }
    }
    if (!pocl) {
      fatal("no pocl platform", "the library", 0);
    }
    cl_int err =
        clGetDeviceIDs(pocl, CL_DEVICE_TYPE_ALL, 1, &c.device, nullptr);
    if (err != CL_SUCCESS) {
      fatal("no device", "pocl", err);
    }
    c.context = clCreateContext(nullptr, 1, &c.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
      fatal("clCreateContext failed", "pocl", err);
    }
    c.queue =
        clCreateCommandQueueWithProperties(c.context, c.device, nullptr, &err);
    if (err != CL_SUCCESS) {
      fatal("clCreateCommandQueueWithProperties failed", "pocl", err);
    }
    // The kernel takes the pointers of each tensor as plain addresses, as on
    // Level Zero. This extension of pocl passes an address with no cl_mem.
    c.set_device_pointer = (clSetKernelArgDevicePointerEXT_fn)
        clGetExtensionFunctionAddressForPlatform(
            pocl, "clSetKernelArgDevicePointerEXT");
    if (!c.set_device_pointer) {
      fatal("no clSetKernelArgDevicePointerEXT", "pocl", 0);
    }
    return c;
  }();
  return c;
}

cl_kernel load(const std::string &key, const char *name, const void *image,
               size_t size) {
  static std::unordered_map<std::string, cl_kernel> cache;
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  // pocl reads SPIR-V up to version 1.2, and the images say 1.4. A Vx image
  // uses nothing that 1.4 added, so the version word is the only change.
  std::vector<uint32_t> words((size + 3) / 4);
  memcpy(words.data(), image, size);
  words[1] = 0x00010200;
  Cl &c = cl();
  cl_int err;
  cl_program program =
      clCreateProgramWithIL(c.context, words.data(), size, &err);
  if (err != CL_SUCCESS) {
    fatal("clCreateProgramWithIL failed", name, err);
  }
  err = clBuildProgram(program, 1, &c.device, "", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[8192] = {0};
    clGetProgramBuildInfo(program, c.device, CL_PROGRAM_BUILD_LOG,
                          sizeof log - 1, log, nullptr);
    fprintf(stderr, "[pocl] the build log of %s:\n%s\n", name, log);
    fatal("clBuildProgram failed", name, err);
  }
  cl_kernel kernel = clCreateKernel(program, name, &err);
  if (err != CL_SUCCESS) {
    fatal("clCreateKernel failed", name, err);
  }
  cache.emplace(key, kernel);
  return kernel;
}

bool run_image(const void *payload, size_t payload_size, void **device_args,
               const int32_t *arg_tags, int64_t num_args) {
  const char *name = static_cast<const char *>(payload);
  const char *format = vx_payload_field(payload, payload_size, "format=");
  if (!format || strcmp(format, "spirv") != 0) {
    return false;
  }
  const void *image = nullptr;
  int64_t image_size = vx_payload_section(payload, payload_size, &image);
  if (image_size <= 0 || !image) {
    return false;
  }

  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);

  vx_launch_params params;
  if (!vx_launch_build_params(device_args, arg_tags, num_args, &params)) {
    return false;
  }
  // A tensor gives two pointers, then its offset, sizes and strides of 8 bytes
  // each. A number gives its own width.
  std::vector<bool> is_pointer;
  std::vector<size_t> width;
  for (int64_t i = 0; i < num_args; ++i) {
    if (VX_ABI_KIND(arg_tags[i]) == VX_ABI_KIND_MEMREF) {
      is_pointer.insert(is_pointer.end(), {true, true});
      is_pointer.insert(is_pointer.end(), 1 + 2 * VX_ABI_RANK(arg_tags[i]),
                        false);
      width.insert(width.end(), 3 + 2 * VX_ABI_RANK(arg_tags[i]), 8);
    } else {
      is_pointer.push_back(false);
      width.push_back(vx_dtype_bytes(VX_ABI_KIND(arg_tags[i])));
    }
  }
  if ((int)is_pointer.size() != params.count) {
    fatal("the parameters and the arguments differ in number", name,
          params.count);
  }

  char library[32];
  snprintf(library, sizeof library, "%p", vx_payload_library(payload));
  cl_kernel kernel =
      load(std::string(name) + "@" + library, name, image, (size_t)image_size);
  const bool flagged = vx_launch_takes_assert_flag(payload, payload_size) != 0;
  cl_uint declared = 0;
  clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof declared, &declared,
                  nullptr);
  if ((int)declared != params.count + (flagged ? 1 : 0)) {
    fatal("the kernel declares another number of parameters", name,
          (int)declared);
  }
  Cl &c = cl();
  for (int i = 0; i < params.count; ++i) {
    cl_int err;
    if (is_pointer[i]) {
      err = c.set_device_pointer(
          kernel, i, (cl_mem_device_address_ext)(*(void **)params.params[i]));
    } else {
      err = clSetKernelArg(kernel, i, width[i], params.params[i]);
    }
    if (err != CL_SUCCESS) {
      fatal("an argument could not be set", name, i);
    }
  }

  // A kernel that can fail an assert takes, last, a flag that a failed assert
  // sets.
  cl_mem flag = nullptr;
  if (flagged) {
    cl_int zero = 0, err = CL_SUCCESS;
    flag = clCreateBuffer(c.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof zero, &zero, &err);
    if (err != CL_SUCCESS || clSetKernelArg(kernel, params.count, sizeof flag,
                                            &flag) != CL_SUCCESS) {
      fatal("the assert flag could not be set", name, err);
    }
  }

  // The launch shape of runtime/xpu_dispatch.cpp.
  unsigned grid[3] = {1, 1, 1};
  unsigned group_size = 1;
  char why[160];
  int set = vx_launch_shape(payload, payload_size, device_args, arg_tags,
                            num_args, grid, &group_size, why, sizeof why);
  if (set < 0) {
    fprintf(stderr, "[pocl] FATAL: %s: %s\n", name, why);
    abort();
  }
  if (!set) {
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
  size_t global[3] = {(size_t)grid[0] * group_size, grid[1], grid[2]};
  size_t local[3] = {group_size, 1, 1};
  cl_int err = clEnqueueNDRangeKernel(c.queue, kernel, 3, nullptr, global,
                                      local, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    fatal("clEnqueueNDRangeKernel failed", name, err);
  }
  clFinish(c.queue);
  if (flag) {
    cl_int failed = 0;
    clEnqueueReadBuffer(c.queue, flag, CL_TRUE, 0, sizeof failed, &failed, 0,
                        nullptr, nullptr);
    clReleaseMemObject(flag);
    if (failed) {
      fflush(stdout);
      fprintf(stderr, "[pocl] FATAL: an assert failed in %s\n", name);
      abort();
    }
  }
  return true;
}

// Printed in every run, also when no kernel ran on pocl.
__attribute__((constructor)) void print_the_counts_at_exit() {
  atexit([] {
    fprintf(
        stderr,
        "[pocl] SPIR-V kernels run on OpenCL: %d, run on the host copy: %d\n",
        g_ran.load(), g_on_host.load());
  });
}

} // namespace

extern "C" uint64_t vx_plugin_dispatch_async(const void *payload,
                                             size_t payload_size,
                                             void **device_args,
                                             const int32_t *arg_tags,
                                             int64_t num_args) {
  if (run_image(payload, payload_size, device_args, arg_tags, num_args)) {
    g_ran++;
    return 1;
  }
  const char *format = vx_payload_field(payload, payload_size, "format=");
  if (format && strcmp(format, "spirv") == 0) {
    g_on_host++;
  }
  return vx_host_dispatch_async(payload, payload_size, device_args, arg_tags,
                                num_args);
}
