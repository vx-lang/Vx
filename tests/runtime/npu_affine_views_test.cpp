//===- npu_affine_views_test.cpp - The affine model and views ---*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Mac dispatcher sends out[i] = a[i] * alpha + beta to the CoreML model
// affine_4, which copies 4 adjacent floats in and 4 out. A vector of 4 with
// stride 1 can use it, also when it starts at an offset. A view with stride 2
// and a transposed 4x4 tensor must run on the CPU. Every case must give the
// right elements.
//
// The CPU kernels below stand in for the kernels the compiler generates. They
// read each tensor through its descriptor, and they count their calls, so the
// test knows which path ran.
//
// Driven by tests/integration_test/npu_affine_views_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../include/vx_hardware_runtime.h"

#include <cstdio>

namespace {

struct Desc1 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

struct Desc2 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

int cpu_calls = 0;
int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

float &at(Desc1 *d, int64_t i) {
  return d->aligned[d->offset + i * d->strides[0]];
}

float &at(Desc2 *d, int64_t i, int64_t j) {
  return d->aligned[d->offset + i * d->strides[0] + j * d->strides[1]];
}

} // namespace

extern "C" void _vx_ciface_vx_npu_kernel_affine(Desc1 *out, Desc1 *a,
                                                float alpha, float beta) {
  ++cpu_calls;
  for (int64_t i = 0; i < out->sizes[0]; ++i) {
    at(out, i) = at(a, i) * alpha + beta;
  }
}

extern "C" void _vx_ciface_vx_npu_kernel_affine_4x4(Desc2 *out, Desc2 *a,
                                                    float alpha, float beta) {
  ++cpu_calls;
  for (int64_t i = 0; i < out->sizes[0]; ++i) {
    for (int64_t j = 0; j < out->sizes[1]; ++j) {
      at(out, i, j) = at(a, i, j) * alpha + beta;
    }
  }
}

namespace {

const char kAffine[] = "vx_npu_kernel_affine\0abi=1\0";
const char kAffine4x4[] = "vx_npu_kernel_affine_4x4\0abi=1\0";

// Dispatches out = a * 2 + 1, and returns how many times a CPU kernel ran.
template <typename Desc>
int dispatch(const char *payload, size_t payload_size, Desc *out, Desc *a,
             int32_t rank) {
  void *out_p = out;
  void *a_p = a;
  float alpha = 2, beta = 1;
  void *args[4] = {&out_p, &a_p, &alpha, &beta};
  const int32_t tags[4] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, rank),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, rank),
                           VX_ABI_KIND_F32, VX_ABI_KIND_F32};
  int before = cpu_calls;
  vx_plugin_dispatch_async(payload, payload_size, args, tags, 4);
  return cpu_calls - before;
}

void test_a_dense_vector_uses_the_model() {
  float a[4] = {0, 1, 2, 3};
  float out[4] = {0, 0, 0, 0};
  Desc1 a_desc = {a, a, 0, {4}, {1}};
  Desc1 out_desc = {out, out, 0, {4}, {1}};
  check(dispatch(kAffine, sizeof(kAffine), &out_desc, &a_desc, 1) == 0,
        "a dense vector of 4 runs on the model, not on the CPU");
  check(out[0] == 1 && out[1] == 3 && out[2] == 5 && out[3] == 7,
        "the model computes 2 * a + 1 for a dense vector");
}

void test_an_input_with_an_offset() {
  // The second row of a 2x4 tensor starts at element 4.
  float a[8] = {0, 1, 2, 3, 10, 11, 12, 13};
  float out[4] = {0, 0, 0, 0};
  Desc1 a_desc = {a, a, 4, {4}, {1}};
  Desc1 out_desc = {out, out, 0, {4}, {1}};
  check(dispatch(kAffine, sizeof(kAffine), &out_desc, &a_desc, 1) == 0,
        "an input with an offset runs on the model");
  check(out[0] == 21 && out[1] == 23 && out[2] == 25 && out[3] == 27,
        "an input with an offset is read from its offset");
}

void test_an_input_with_stride_2() {
  float a[8] = {0, 100, 1, 100, 2, 100, 3, 100};
  float out[4] = {0, 0, 0, 0};
  Desc1 a_desc = {a, a, 0, {4}, {2}};
  Desc1 out_desc = {out, out, 0, {4}, {1}};
  check(dispatch(kAffine, sizeof(kAffine), &out_desc, &a_desc, 1) == 1,
        "an input with stride 2 runs on the CPU");
  check(out[0] == 1 && out[1] == 3 && out[2] == 5 && out[3] == 7,
        "an input with stride 2 reads every other element");
}

void test_an_output_with_an_offset() {
  float a[4] = {0, 1, 2, 3};
  float out[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  Desc1 a_desc = {a, a, 0, {4}, {1}};
  Desc1 out_desc = {out, out, 4, {4}, {1}};
  check(dispatch(kAffine, sizeof(kAffine), &out_desc, &a_desc, 1) == 0,
        "an output with an offset runs on the model");
  check(out[0] == -1 && out[1] == -1 && out[2] == -1 && out[3] == -1,
        "the elements before an output's offset are unchanged");
  check(out[4] == 1 && out[5] == 3 && out[6] == 5 && out[7] == 7,
        "an output with an offset is written at its offset");
}

void test_a_transposed_4x4_tensor() {
  // Strides (1, 4): the first dimension has stride 1, as in a vector of 4.
  float a[16];
  float out[16];
  for (int i = 0; i < 16; ++i) {
    a[i] = (float)i;
    out[i] = 0;
  }
  Desc2 a_desc = {a, a, 0, {4, 4}, {1, 4}};
  Desc2 out_desc = {out, out, 0, {4, 4}, {1, 4}};
  check(dispatch(kAffine4x4, sizeof(kAffine4x4), &out_desc, &a_desc, 2) == 1,
        "a transposed 4x4 tensor runs on the CPU");
  bool all = true;
  for (int i = 0; i < 16; ++i) {
    all = all && out[i] == (float)(2 * i + 1);
  }
  check(all, "all 16 elements of a transposed 4x4 tensor are computed");
}

} // namespace

int main() {
  test_a_dense_vector_uses_the_model();
  test_an_input_with_an_offset();
  test_an_input_with_stride_2();
  test_an_output_with_an_offset();
  test_a_transposed_4x4_tensor();
  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
