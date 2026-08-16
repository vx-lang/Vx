//===- launch_smem_kernel.cpp - Run the first shared-memory kernel -*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Loads the PTX emitted for tests/backend/pass/custom_topology_device_image.vx
// and launches it on a real GPU. This is the first kernel Vx emits that USES
// SHARED MEMORY -- the tile placed in Memory::SMEM materialises as `.shared`,
// filled by a copy and read back by the body (Vx#352, #353) -- and it has never
// executed on silicon. Everything known about it so far comes from reading PTX.
//
// Mirrors scripts/launch_emitted_kernel.cpp deliberately: standalone driver-API
// harness, no plugin, no wire, no worker. A wrong answer here is the kernel or
// the marshalling, and there is no third possibility.
//
// What this verifies, in order:
//   1. `cuModuleLoadData` accepts an image containing a `.shared` declaration
//      and a dynamic-shared-capable entry (the flash kernel had neither);
//   2. `vx_launch_entry_param_count` agrees with the marshaller at 14 -- this
//      kernel is that check's first customer besides the flash kernel's 28;
//   3. the launch succeeds single-threaded. The copy loop has no barrier (the
//      C3 gap, owned by #353 A3) -- SAFE here because one thread cannot race
//      itself, and this run is the empirical confirmation of that claim;
//   4. the arithmetic: o[i][d] must equal (i+d) EXACTLY. Inputs are
//      (i+d)*0.5 and the kernel doubles them; small integers are exact in f32,
//      so any deviation at all is a real defect, not rounding.
//
// One thread (1x1 launch) is the CORRECT launch for this kernel, not a
// shortcut: the region has no thread indexing (#251 defers parallelism), and a
// wide launch would run the whole computation N times over the same output.
//
// Build (driver API + a toolkit's headers, not nvcc):
//
//   g++ -std=c++17 -O2 -I/usr/local/cuda/include \
//       utils/memalg/launch_smem_kernel.cpp \
//       -L/usr/local/cuda/lib64/stubs -lcuda -o launch_smem_kernel
//   ./launch_smem_kernel smem_kernel.ptx vx_npu_kernel_0
//
// The PTX is extracted from `vxc --emit-llvm` by run_a100_session.sh on the
// machine that has the compiler, so the pod needs only this file and the .ptx.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_kernel_launch.h"

#include <cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void must(CUresult r, const char *what) {
  if (r != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *desc = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    fprintf(stderr, "FATAL: %s: %s (%s)\n", what, name ? name : "?",
            desc ? desc : "?");
    exit(1);
  }
}

struct Desc2D {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

Desc2D describe(CUdeviceptr p, int64_t rows, int64_t cols) {
  Desc2D d{};
  d.allocated = (void *)(uintptr_t)p;
  d.aligned = (void *)(uintptr_t)p;
  d.offset = 0;
  d.sizes[0] = rows;
  d.sizes[1] = cols;
  d.strides[0] = cols;
  d.strides[1] = 1;
  return d;
}

std::string read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "FATAL: cannot open %s\n", path);
    exit(1);
  }
  std::string s;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
    s.append(buf, n);
  }
  fclose(f);
  return s;
}

} // namespace

int main(int argc, char **argv) {
  const char *ptx_path = argc > 1 ? argv[1] : "smem_kernel.ptx";
  const char *entry = argc > 2 ? argv[2] : "vx_npu_kernel_0";

  const std::string ptx = read_file(ptx_path);

  // The image must actually contain shared memory, or this harness is
  // verifying the wrong artifact -- the exact mislabelled-cell failure the
  // measurement campaign kept finding, applied to a binary.
  if (ptx.find(".shared") == std::string::npos) {
    fprintf(stderr,
            "FATAL: %s contains no .shared -- this is not the shared-memory "
            "kernel this harness exists to verify\n",
            ptx_path);
    return 1;
  }

  const int declared = vx_launch_entry_param_count(ptx.c_str(), entry);
  if (declared < 0) {
    fprintf(stderr, "FATAL: %s declares no entry named %s\n", ptx_path, entry);
    return 1;
  }

  // The fixture's shapes: two 2x2 f32 tensors, a filled with (i+d)*0.5, o
  // zeroed. Two rank-2 memrefs explode to 2 * (3 + 2*2) = 14 parameters.
  const int R = 2, C = 2;
  std::vector<float> a(R * C), o(R * C, 0.0f);
  for (int i = 0; i < R; ++i)
    for (int d = 0; d < C; ++d)
      a[i * C + d] = (float)(i + d) * 0.5f;

  must(cuInit(0), "cuInit");
  CUdevice dev;
  must(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  must(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

  char name[256] = {0};
  cuDeviceGetName(name, sizeof(name), dev);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       dev);
  printf("device      %s (sm_%d%d)\n", name, major, minor);
  printf("image       %s, %zu bytes, entry %s with %d parameters\n", ptx_path,
         ptx.size(), entry, declared);

  CUmodule mod;
  must(cuModuleLoadData(&mod, ptx.c_str()), "cuModuleLoadData");
  CUfunction fn;
  must(cuModuleGetFunction(&fn, mod, entry), "cuModuleGetFunction");

  CUdeviceptr da, dobuf;
  must(cuMemAlloc(&da, a.size() * 4), "cuMemAlloc a");
  must(cuMemAlloc(&dobuf, o.size() * 4), "cuMemAlloc o");
  must(cuMemcpyHtoD(da, a.data(), a.size() * 4), "H2D a");
  must(cuMemcpyHtoD(dobuf, o.data(), o.size() * 4), "H2D o");

  // Capture order is order of first use in the region: the SMEM transfer reads
  // `ad` first, then the loop writes `od`. A swap here fails loudly rather
  // than plausibly -- the expected output is nonzero and o was zeroed, so
  // reading o and writing a yields all zeros, not a wrong-but-nonzero grid.
  Desc2D dad = describe(da, R, C);
  Desc2D dod = describe(dobuf, R, C);
  void *descs[2] = {&dad, &dod};
  void *device_args[2] = {&descs[0], &descs[1]};
  int32_t tags[2] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
                     VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};

  VxLaunchParams params;
  if (!vx_launch_build_params(&params, device_args, tags, 2)) {
    fprintf(stderr, "FATAL: vx_launch_build_params refused the arguments\n");
    return 1;
  }
  if (params.count != declared) {
    fprintf(stderr,
            "FATAL: marshalled %d parameters but the PTX declares %d -- the "
            "signature and the marshalling disagree\n",
            params.count, declared);
    return 1;
  }

  must(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, params.params,
                      nullptr),
       "cuLaunchKernel");
  must(cuCtxSynchronize(), "cuCtxSynchronize");

  must(cuMemcpyDtoH(o.data(), dobuf, o.size() * 4), "D2H o");

  // Exact comparison, deliberately: (i+d)*0.5*2 = i+d, and 0, 1, 2 are exact
  // in f32. An epsilon here would hide a real defect behind tolerance.
  int bad = 0;
  for (int i = 0; i < R; ++i) {
    for (int d = 0; d < C; ++d) {
      const float want = (float)(i + d);
      const float got = o[i * C + d];
      if (got != want) {
        fprintf(stderr, "MISMATCH o[%d][%d]: got %.9g want %.9g\n", i, d, got,
                want);
        ++bad;
      }
    }
  }
  if (bad) {
    fprintf(stderr, "FAILED: %d of %d elements wrong\n", bad, R * C);
    return 1;
  }
  printf("OK: %d elements exact; the shared-memory kernel computes what the "
         "host computes\n",
         R * C);
  return 0;
}
