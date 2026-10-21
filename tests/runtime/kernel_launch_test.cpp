//===- kernel_launch_test.cpp - Marshalling to a kernel ---------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exercises runtime/vx_kernel_launch.h, which turns a dispatch's arguments into
// the parameter list a device entry point expects (#251).
//
// Worth testing away from a GPU because the failure is silent. A parameter list
// that is the wrong length or the wrong order still launches: `cuLaunchKernel`
// cannot check it against the kernel's signature, so the kernel reads its
// arguments from the wrong offsets and computes a plausible wrong answer, or
// reads a size out of a pointer field and walks off the buffer. Neither shows
// up as a failed launch.
//
// The expected counts are pinned against a real kernel: the placed region in
// tests/backend/pass/placed_kernel_four_operands.vx compiles to an entry with
// 28 parameters -- four rank-2 memrefs at seven each -- which
// tests/integration_test/device_image_test.rs reads off the emitted PTX and
// this file asserts from the ABI side.
//
// Driven by tests/integration_test/kernel_launch_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_kernel_launch.h"
#include "../../runtime/vx_stream_policy.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/// A ranked memref descriptor, laid out as MLIR's C interface passes it.
template <int Rank> struct Descriptor {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

using Desc2D = Descriptor<2>;

Desc2D make_2d(void *data, int64_t rows, int64_t cols) {
  Desc2D d{};
  d.allocated = data;
  d.aligned = data;
  d.offset = 0;
  d.sizes[0] = rows;
  d.sizes[1] = cols;
  d.strides[0] = cols;
  d.strides[1] = 1;
  return d;
}

/// The parameters are addresses into the caller's descriptors, so reading one
/// back is how a launch would read it.
int64_t as_i64(void *p) { return *(const int64_t *)p; }
void *as_ptr(void *p) { return *(void *const *)p; }

void widths() {
  check(vx_launch_param_width(VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)) == 7,
        "a rank-2 memref is 2 pointers + offset + 2 sizes + 2 strides");
  check(vx_launch_param_width(VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)) == 5,
        "a rank-1 memref is five parameters");
  check(vx_launch_param_width(VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 0)) == 3,
        "a rank-0 memref is still allocated, aligned and offset");
  check(vx_launch_param_width(VX_ABI_KIND_F32) == 1, "a scalar is one");

  /* A slot has no descriptor yet -- refusing is the point, see #251. */
  check(vx_launch_param_width(VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2)) == 0,
        "a publication slot cannot be marshalled");
}

/// Four rank-2 f32 memrefs, the signature a four-operand placed kernel has.
void the_four_operand_signature() {
  float qs[32 * 16], ks[64 * 16], os[32 * 16], vs[64 * 16];
  Desc2D q = make_2d(qs, 32, 16);
  Desc2D k = make_2d(ks, 64, 16);
  Desc2D o = make_2d(os, 32, 16);
  Desc2D v = make_2d(vs, 64, 16);

  void *descs[4] = {&q, &k, &o, &v};
  void *device_args[4] = {&descs[0], &descs[1], &descs[2], &descs[3]};
  int32_t tags[4] = {
      VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2), VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
      VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2), VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};

  vx_launch_params p;
  check(vx_launch_build_params(device_args, tags, 4, &p),
        "four rank-2 memrefs marshal");
  check(p.count == 28,
        "four rank-2 memrefs are 28 parameters, as the PTX declares");

  /* The first operand, field by field, in the order the entry declares them. */
  check(as_ptr(p.params[0]) == qs, "param 0 is the allocated pointer");
  check(as_ptr(p.params[1]) == qs, "param 1 is the aligned pointer");
  check(as_i64(p.params[2]) == 0, "param 2 is the offset");
  check(as_i64(p.params[3]) == 32, "param 3 is size[0]");
  check(as_i64(p.params[4]) == 16, "param 4 is size[1]");
  check(as_i64(p.params[5]) == 16, "param 5 is stride[0]");
  check(as_i64(p.params[6]) == 1, "param 6 is stride[1]");

  /* And the second starts where the first ended, rather than overlapping it. */
  check(as_ptr(p.params[7]) == ks, "the second operand starts at param 7");
  check(as_i64(p.params[10]) == 64, "the second operand's size[0] is its own");

  /* Nothing was copied: a parameter is an address inside the caller's
     descriptor, so a descriptor edited after marshalling is seen by the
     launch. This is what makes the list free to build, and it is also what
     makes a dangling descriptor fatal, so it should be stated somewhere that
     fails when it stops being true. */
  q.sizes[0] = 99;
  check(as_i64(p.params[3]) == 99, "parameters alias the descriptor");
}

void a_view_keeps_its_own_offset_and_strides() {
  float base[64 * 16];
  Desc2D v = make_2d(base, 8, 16);
  v.offset = 16 * 4; /* four rows in */
  v.strides[0] = 16;

  void *desc = &v;
  void *device_args[1] = {&desc};
  int32_t tags[1] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};

  vx_launch_params p;
  check(vx_launch_build_params(device_args, tags, 1, &p), "a view marshals");
  check(p.count == 7, "a view is still seven parameters");
  check(as_ptr(p.params[0]) == base,
        "a view's pointer is the base, not the first element -- the offset is "
        "a separate parameter and adding it here would apply it twice");
  check(as_i64(p.params[2]) == 64, "the offset travels as itself");
  check(as_i64(p.params[5]) == 16, "and the row stride with it");
}

void scalars_pass_by_address() {
  int64_t n = 7;
  float alpha = 0.5f;
  void *device_args[2] = {&n, &alpha};
  int32_t tags[2] = {VX_ABI_KIND_I64, VX_ABI_KIND_F32};

  vx_launch_params p;
  check(vx_launch_build_params(device_args, tags, 2, &p), "scalars marshal");
  check(p.count == 2, "one parameter each");
  check(p.params[0] == &n && p.params[1] == &alpha,
        "a scalar parameter is the address of the value");
}

/// Every refusal, because each one is a launch that must not happen.
void refusals() {
  vx_launch_params p;

  {
    Desc2D d = make_2d(nullptr, 1, 1);
    void *desc = &d;
    void *device_args[1] = {&desc};
    int32_t tags[1] = {VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2)};
    check(!vx_launch_build_params(device_args, tags, 1, &p),
          "a slot argument is refused rather than marshalled as a pointer");
  }
  {
    void *nothing = nullptr;
    void *device_args[1] = {&nothing};
    int32_t tags[1] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};
    check(!vx_launch_build_params(device_args, tags, 1, &p),
          "an argument holding no descriptor is refused");
  }
  {
    void *device_args[1] = {nullptr};
    int32_t tags[1] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};
    check(!vx_launch_build_params(device_args, tags, 1, &p),
          "a null argument slot is refused");
  }
  {
    /* Twenty rank-3 memrefs is 180 parameters, past the cap. The refusal has
       to be whole: a list truncated at the cap launches, and the kernel reads
       the parameters that were never written. */
    Desc2D d = make_2d(nullptr, 1, 1);
    void *desc = &d;
    void *device_args[32];
    int32_t tags[32];
    for (int i = 0; i < 32; ++i) {
      device_args[i] = &desc;
      tags[i] = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 3);
    }
    check(!vx_launch_build_params(device_args, tags, 32, &p),
          "a parameter list that would not fit is refused entirely");
    check(p.count == 0, "and reports no parameters rather than a prefix");
  }
}

/// The signature check that stands between a mismatch and a silent wrong
/// answer.
void entry_parameter_counting() {
  static const char kPtx[] =
      "//\n// Generated by LLVM NVPTX Back-End\n//\n\n"
      ".version 7.6\n.target sm_80\n.address_size 64\n\n"
      "\t// .globl\tvx_npu_kernel_0\n\n"
      ".visible .entry vx_npu_kernel_0(\n"
      "\t.param .u64 .ptr .align 1 vx_npu_kernel_0_param_0,\n"
      "\t.param .u64 vx_npu_kernel_0_param_1,\n"
      "\t.param .u64 vx_npu_kernel_0_param_2\n"
      ")\n"
      "{\n"
      "\t.reg .b64 \t%rd<4>;\n"
      "\tld.param.u64 \t%rd1, [vx_npu_kernel_0_param_0];\n"
      "\tret;\n"
      "}\n";

  check(vx_launch_entry_param_count(kPtx, "vx_npu_kernel_0") == 3,
        "the signature is counted, not every .param in the file");
  check(vx_launch_entry_param_count(kPtx, "vx_npu_kernel_1") == -1,
        "an entry that is not there is reported as absent");

  /* A prefix must not answer for a longer name, which starts mattering at the
     eleventh outlined region. */
  static const char kTwo[] = ".visible .entry vx_npu_kernel_10(\n"
                             "\t.param .u64 a,\n"
                             "\t.param .u64 b\n"
                             ")\n{\nret;\n}\n";
  check(vx_launch_entry_param_count(kTwo, "vx_npu_kernel_1") == -1,
        "`vx_npu_kernel_1` does not match `vx_npu_kernel_10`");
  check(vx_launch_entry_param_count(kTwo, "vx_npu_kernel_10") == 2,
        "but the full name does");

  check(vx_launch_entry_param_count(kPtx, nullptr) == -1, "no entry, no count");
  check(vx_launch_entry_param_count(nullptr, "x") == -1, "no PTX, no count");
}

} // namespace

/// `grid([..]) block(n)` (#252): the payload names the sizes. A size `$j` is
/// the `i32` launch value j, which comes after the kernel's arguments.
void launch_shapes() {
  int32_t rows = 7;
  int64_t wide = 7;
  float data[4] = {};
  void *ptr = data;
  /* One kernel argument, then the launch values. */
  void *args[3] = {&ptr, &rows, &wide};
  const int32_t tags[3] = {0, 4, 5};
  unsigned grid[3] = {0, 0, 0};
  unsigned block = 0;
  char why[160] = "";

  const char shaped[] = "k\0grid=$0,2\0block=128\0";
  check(vx_launch_shape(shaped, sizeof shaped, args, tags, 1, grid, &block, why,
                        sizeof why) == 1,
        "a payload with grid= and block= has a shape");
  check(grid[0] == 7 && grid[1] == 2 && grid[2] == 1,
        "the grid is launch value 0, then 2, then 1 for the missing axis");
  check(block == 128, "the block is the number in block=");

  const char plain[] = "k\0launch=64\0";
  check(vx_launch_shape(plain, sizeof plain, args, tags, 1, grid, &block, why,
                        sizeof why) == 0,
        "a payload without grid= has no shape");

  const char big_block[] = "k\0grid=1\0block=2048\0";
  check(vx_launch_shape(big_block, sizeof big_block, args, tags, 1, grid,
                        &block, why, sizeof why) == -1 &&
            strstr(why, "1 to 1024 threads"),
        "a block of more than 1024 threads is refused");

  const char empty_grid[] = "k\0grid=0\0block=32\0";
  check(vx_launch_shape(empty_grid, sizeof empty_grid, args, tags, 1, grid,
                        &block, why, sizeof why) == -1 &&
            strstr(why, "grid axis 0 has 0 blocks"),
        "a grid axis with no blocks is refused");

  const char not_i32[] = "k\0grid=$1\0block=32\0";
  check(vx_launch_shape(not_i32, sizeof not_i32, args, tags, 1, grid, &block,
                        why, sizeof why) == -1,
        "a launch value that is not an i32 is refused");
}

/* With a stream set by the host program, a wait covers that stream, and a
   graph recording refuses a wait. Memory from cudaMallocAsync is freed in
   stream order, so a recording can hold its free. */
void caller_streams() {
  check(vx_wait_action_for(0, 0) == VX_WAIT_DEVICE,
        "with no stream, a wait covers the device");
  check(vx_wait_action_for(1, 0) == VX_WAIT_STREAM,
        "with a stream, a wait covers that stream");
  check(vx_wait_action_for(1, 1) == VX_WAIT_REFUSE,
        "a recording refuses a wait");
  check(vx_free_action_for(0, 0, 0) == VX_FREE_AFTER_DEVICE_WAIT,
        "with no stream, a free waits for the device");
  check(vx_free_action_for(0, 0, 1) == VX_FREE_ON_STREAM,
        "stream-ordered memory goes back on its stream after a reset to null");
  check(vx_free_action_for(1, 1, 1) == VX_FREE_ON_STREAM,
        "a recording holds the free of stream-ordered memory");
  check(vx_free_action_for(1, 0, 1) == VX_FREE_ON_STREAM,
        "stream-ordered memory is freed with no wait");
  check(vx_free_action_for(1, 0, 0) == VX_FREE_AFTER_STREAM_WAIT,
        "other memory waits for the stream before its free");
  check(vx_free_action_for(1, 1, 0) == VX_FREE_REFUSE,
        "a recording refuses a free that needs a wait");
}

int main() {
  widths();
  the_four_operand_signature();
  a_view_keeps_its_own_offset_and_strides();
  scalars_pass_by_address();
  refusals();
  entry_parameter_counting();
  launch_shapes();
  caller_streams();

  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all kernel launch marshalling checks passed\n");
  return 0;
}
