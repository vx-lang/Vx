//===- launch_emitted_kernel.cpp - Run the compiler's own kernel -*- C++
//-*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Loads the PTX the compiler put in a dispatch payload and launches it on a
// real GPU, with arguments marshalled by runtime/vx_kernel_launch.h -- the same
// header runtime/cuda_dispatch.cpp will use (#251).
//
// This is the step between "the kernel reaches PTX" and "a dispatch runs it".
// Everything up to here was checked without a device: the pipeline agreed with
// scripts/flash_kernel_to_ptx.sh byte for byte, and the parameter list agreed
// with the signature at 28. Neither says the kernel *computes* anything, and
// both would look exactly the same if it did not.
//
// So this checks the arithmetic, against a number that was established
// independently. tests/backend/pass/flash_attention_placed_verified.vx carries
// `EXPECT: [[0.901606, 0.941606, 0.981606` -- row 0 of a 32x16 fused attention
// output, checked element by element against a separate implementation when
// that file was written (worst disagreement 5.08e-06). This program builds the
// same inputs, launches the compiler's kernel on them, and compares.
//
// Deliberately not a Vx program and not a dispatch. If the kernel is wrong,
// this says so with nothing else in the frame -- no plugin, no wire, no worker.
// A wrong answer here is the kernel or the marshalling, and there is no third
// possibility.
//
// One thread. The region has no thread indexing in it -- one thread runs all 32
// queries -- which is a deliberate omission (#251 defers the parallelism), so a
// 1x1 launch is the *correct* launch for this kernel rather than a shortcut.
// Launching it wide would run the same whole computation 32 times over the same
// output and race.
//
// Build (needs the CUDA driver API and a toolkit's headers, not nvcc):
//
//   g++ -std=c++17 -O2 -I/usr/local/cuda/include
//       scripts/launch_emitted_kernel.cpp
//       -L/usr/local/cuda/lib64/stubs -lcuda -o launch_emitted_kernel
//   ./launch_emitted_kernel kernel.ptx vx_npu_kernel_0
//
// The PTX comes out of a dispatch payload; scripts/flash_kernel_to_ptx.sh
// writes one out, or extract the `image=` field from `vxc --emit-llvm`. Link
// libdevice when producing it (VX_LIBDEVICE), or `__nv_expf` is unresolved and
// cuModuleLoadData refuses the image.
//
//===----------------------------------------------------------------------===//

#include "../runtime/vx_kernel_launch.h"

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

/// A ranked memref descriptor, laid out as MLIR's C interface passes it. The
/// pointers are device addresses; the descriptor itself stays on the host,
/// because a launch copies parameter *values* into the kernel's parameter
/// space.
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
  const char *ptx_path = argc > 1 ? argv[1] : "kernel.ptx";
  const char *entry = argc > 2 ? argv[2] : "vx_npu_kernel_0";

  const std::string ptx = read_file(ptx_path);

  // The signature check, before anything is launched. Without it a mismatch is
  // silent: the driver cannot size-check a parameter array, so the kernel would
  // read past what was supplied.
  const int declared = vx_launch_entry_param_count(ptx.c_str(), entry);
  if (declared < 0) {
    fprintf(stderr, "FATAL: %s declares no entry named %s\n", ptx_path, entry);
    return 1;
  }

  // The shapes tests/backend/pass/flash_attention_placed_verified.vx uses: 32
  // queries over 64 keys at head dimension 16.
  const int Q = 32, K = 64, D = 16;
  std::vector<float> q(Q * D), k(K * D), v(K * D), o(Q * D, 0.0f);
  for (int i = 0; i < Q; ++i) {
    for (int d = 0; d < D; ++d) {
      q[i * D + d] = (float)(i + d) * 0.05f;
    }
  }
  for (int j = 0; j < K; ++j) {
    for (int d = 0; d < D; ++d) {
      k[j * D + d] = (float)(j - d) * 0.03f;
      v[j * D + d] = (float)(j + 2 * d) * 0.02f;
    }
  }

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

  // The driver JITs the PTX here. This is the step that makes shipping text
  // rather than a cubin worth the cost: no ptxas anywhere in the pipeline, and
  // the image stays loadable on a newer device.
  CUmodule mod;
  must(cuModuleLoadData(&mod, ptx.c_str()), "cuModuleLoadData");
  CUfunction fn;
  must(cuModuleGetFunction(&fn, mod, entry), "cuModuleGetFunction");

  CUdeviceptr dq, dk, dv, dobuf;
  must(cuMemAlloc(&dq, q.size() * 4), "cuMemAlloc q");
  must(cuMemAlloc(&dk, k.size() * 4), "cuMemAlloc k");
  must(cuMemAlloc(&dv, v.size() * 4), "cuMemAlloc v");
  must(cuMemAlloc(&dobuf, o.size() * 4), "cuMemAlloc o");
  must(cuMemcpyHtoD(dq, q.data(), q.size() * 4), "H2D q");
  must(cuMemcpyHtoD(dk, k.data(), k.size() * 4), "H2D k");
  must(cuMemcpyHtoD(dv, v.data(), v.size() * 4), "H2D v");
  // Zeroed on the device, because the program transfers an initialised `o_h`
  // and the kernel accumulates into it rather than overwriting it.
  must(cuMemcpyHtoD(dobuf, o.data(), o.size() * 4), "H2D o");

  // The capture order the outliner produced, which is order of first use:
  // q, k, o, v. Reading it off `vx.launch @vx_npu_kernel_0(%56, %57, %59, %58)`
  // rather than guessing -- the two 32x16 operands are q and o, and swapping
  // them would produce a plausible wrong answer rather than a failure.
  Desc2D dqd = describe(dq, Q, D);
  Desc2D dkd = describe(dk, K, D);
  Desc2D dod = describe(dobuf, Q, D);
  Desc2D dvd = describe(dv, K, D);

  void *descs[4] = {&dqd, &dkd, &dod, &dvd};
  void *device_args[4] = {&descs[0], &descs[1], &descs[2], &descs[3]};
  int32_t tags[4] = {
      VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2), VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
      VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2), VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};

  vx_launch_params params;
  if (!vx_launch_build_params(device_args, tags, 4, &params)) {
    fprintf(stderr, "FATAL: the arguments could not be marshalled\n");
    return 1;
  }
  if (params.count != declared) {
    fprintf(
        stderr,
        "FATAL: the kernel declares %d parameters and the arguments "
        "produced %d. Launching anyway would read past what was supplied.\n",
        declared, params.count);
    return 1;
  }
  printf("marshalled  %d parameters, matching the signature\n", params.count);

  must(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, params.params, nullptr),
       "cuLaunchKernel");
  must(cuCtxSynchronize(), "cuCtxSynchronize");
  must(cuMemcpyDtoH(o.data(), dobuf, o.size() * 4), "D2H o");

  printf("row 0       ");
  for (int d = 0; d < 6; ++d) {
    printf("%.6f%s", o[d], d + 1 < 6 ? ", " : " ...\n");
  }

  // The pinned row from the .vx file's EXPECT. The tolerance is f32 summation
  // order, which is what separated a GPU run from a local one by 2 ulp when the
  // placement work measured it (17.068802 against 17.068798) -- not an
  // algorithm difference, and not something to hide behind a loose bound
  // either.
  static const float kExpected[3] = {0.901606f, 0.941606f, 0.981606f};
  int bad = 0;
  for (int d = 0; d < 3; ++d) {
    const float diff = std::fabs(o[d] - kExpected[d]);
    if (!(diff < 1e-5f)) {
      fprintf(stderr, "FAIL: row 0 element %d is %.6f, expected %.6f (%.2e)\n",
              d, o[d], kExpected[d], diff);
      ++bad;
    }
  }

  // Row 1 too. Row 0 and row 1 are close enough to be confused (0.901606
  // against 0.928645), and a kernel that wrote the same row 32 times would pass
  // a row-0-only check.
  printf("row 1       %.6f\n", o[D]);
  if (!(std::fabs(o[D] - 0.928645f) < 1e-5f)) {
    fprintf(stderr,
            "FAIL: row 1 is %.6f, expected 0.928645 -- every row must "
            "be its own\n",
            o[D]);
    ++bad;
  }

  cuMemFree(dq);
  cuMemFree(dk);
  cuMemFree(dv);
  cuMemFree(dobuf);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);

  if (bad != 0) {
    fprintf(stderr, "%d check(s) failed\n", bad);
    return 1;
  }
  printf("\nthe compiler's own kernel ran on the device and agrees with the "
         "verified output\n");
  return 0;
}
