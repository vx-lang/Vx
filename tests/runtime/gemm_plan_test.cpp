//===- gemm_plan_test.cpp - Decode tests for the dispatch plan --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exercises runtime/vx_dispatch_plan.h against arguments built the way the
// compiler builds them, on any machine: the decode decides which buffer a
// vendor GEMM reads and which it writes, and getting it wrong produces a
// plausible matrix of wrong numbers rather than a failure. That is not
// something to discover on a rented GPU.
//
// The payload blobs below are written out by hand rather than produced by the
// compiler, which is deliberate -- it pins the wire format that
// src/dialect/VxLowering.cpp emits and tests/optimizations/pass/
// kernel_kind_matmul.vx checks, so a change to either side without the other
// fails here.
//
// Driven by tests/integration_test/gemm_plan_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_dispatch_plan.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/// A ranked memref descriptor, laid out as MLIR's C-interface passes it:
/// {allocated, aligned, offset, sizes[rank], strides[rank]}.
template <int Rank> struct Descriptor {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

using Desc2D = Descriptor<2>;

Desc2D make_2d(void *data, int64_t rows, int64_t cols, int64_t row_stride) {
  Desc2D d{};
  d.allocated = data;
  d.aligned = data;
  d.offset = 0;
  d.sizes[0] = rows;
  d.sizes[1] = cols;
  d.strides[0] = row_stride;
  d.strides[1] = 1;
  return d;
}

/// A NUL-separated payload: kernel name first, then `key=value` entries. The
/// name leading is what lets a consumer predating the extension read the blob
/// as a plain C string and still get the name.
std::string payload_of(const char *name, const char *kind, const char *roles,
                       const char *outkind) {
  std::string p(name);
  p.push_back('\0');
  if (kind) {
    p += "kind=";
    p += kind;
    p.push_back('\0');
  }
  if (roles) {
    p += "roles=";
    p += roles;
    p.push_back('\0');
  }
  if (outkind) {
    p += "outkind=";
    p += outkind;
    p.push_back('\0');
  }
  return p;
}

void test_parse_roles() {
  int a = -1, b = -1, o = -1;

  check(vx_parse_roles("a:0,b:2,out:5", &a, &b, &o) && a == 0 && b == 2 &&
            o == 5,
        "roles in the order the compiler emits them");

  a = b = o = -1;
  check(vx_parse_roles("out:1,a:7,b:3", &a, &b, &o) && a == 7 && b == 3 &&
            o == 1,
        "roles in any order");

  // A partial mapping invites the reader to supply the rest by convention,
  // which is the guessing the roles exist to replace.
  check(!vx_parse_roles("a:0,b:2", &a, &b, &o), "a missing role is a refusal");
  check(!vx_parse_roles("a:0,b:2,out:5,c:6", &a, &b, &o), "unknown role");
  check(!vx_parse_roles("a:0,b:2,out:", &a, &b, &o), "missing index");
  check(!vx_parse_roles("a:0,b:2,out:-1", &a, &b, &o), "negative index");
  check(!vx_parse_roles("a:0,b:2,out:5,", &a, &b, &o), "trailing comma");
  check(!vx_parse_roles("a:0,a:1,b:2,out:5", &a, &b, &o), "a role named twice");
  check(!vx_parse_roles("", &a, &b, &o), "empty roles");
  check(!vx_parse_roles(nullptr, &a, &b, &o), "absent roles");
}

/// A rank-0 memref: storage holding a descriptor, which is how a tensor
/// declared inside a function is captured.
struct SlotDesc {
  void *allocated;
  void *aligned;
  int64_t offset;
};

/// The shape the compiler actually emits for `c = a @ b`: six launch operands,
/// of which 0 and 2 are the inputs and 5 is the slot the kernel publishes
/// through. The others are loop bounds and the fill constant.
struct SlotCase {
  std::vector<float> a_data;
  std::vector<float> b_data;
  Desc2D a_desc;
  Desc2D b_desc;
  Desc2D result_desc; // storage the result slot points at
  SlotDesc slot_desc;
  SlotDesc a_slot_desc;
  SlotDesc b_slot_desc;

  int64_t idx0 = 0, idx1 = 1;
  float fill = 0.0f;

  void *a_ptr;
  void *b_ptr;
  void *slot_ptr;
  void *a_slot_ptr;
  void *b_slot_ptr;
  void *i0_ptr;
  void *i1_ptr;
  void *fill_ptr;

  std::vector<void *> args;
  std::vector<int32_t> tags;

  SlotCase(int64_t m, int64_t k, int64_t n)
      : a_data((size_t)(m * k), 1.0f), b_data((size_t)(k * n), 1.0f) {
    a_desc = make_2d(a_data.data(), m, k, k);
    b_desc = make_2d(b_data.data(), k, n, n);
    result_desc = make_2d(nullptr, 0, 0, 0);
    slot_desc = {&result_desc, &result_desc, 0};

    a_ptr = &a_desc;
    b_ptr = &b_desc;
    slot_ptr = &slot_desc;
    i0_ptr = &idx0;
    i1_ptr = &idx1;
    fill_ptr = &fill;

    args = {&a_ptr, i0_ptr, &b_ptr, i1_ptr, fill_ptr, &slot_ptr};
    tags = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
            VX_ABI_KIND_I64,
            VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
            VX_ABI_KIND_I64,
            VX_ABI_KIND_F32,
            VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2)};
  }

  /// Present the inputs the way local tensors arrive: captured as the slots
  /// they live in rather than as the buffers themselves. Only a function
  /// parameter is captured directly, so this is the common case, not the
  /// exotic one.
  void inputs_as_locals() {
    a_slot_desc = {&a_desc, &a_desc, 0};
    b_slot_desc = {&b_desc, &b_desc, 0};
    a_slot_ptr = &a_slot_desc;
    b_slot_ptr = &b_slot_desc;
    args[0] = &a_slot_ptr;
    args[2] = &b_slot_ptr;
    tags[0] = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);
    tags[2] = VX_ABI_SLOT_TAG(VX_DTYPE_F32, 2);
  }
};

/// The device a launch targets rides in the same payload, and a plugin turns
/// the topology id into a device ordinal. Getting this wrong in a disaggregated
/// run means the right answer computed on the wrong half of the machine.
void test_topology() {
  std::string p0 = payload_of("k", "matmul", "a:0,b:2,out:5", "slot") +
                   std::string("topo=500\0", 10);
  std::string p1 = payload_of("k", "matmul", "a:0,b:2,out:5", "slot") +
                   std::string("topo=501\0", 10);

  check(vx_payload_topology(p0.data(), p0.size()) == 500, "topo= is read back");
  check(vx_payload_topology(p1.data(), p1.size()) == 501, "a second device");

  // Absent means a producer predating the entry, which is not device 0: a
  // plugin that conflated them would set the device from nothing at all.
  std::string none = payload_of("k", "matmul", "a:0,b:2,out:5", "slot");
  check(vx_payload_topology(none.data(), none.size()) == -1, "absent is not 0");

  check(vx_topology_device_index(500, VX_TOPO_GPU_BASE) == 0, "GPU[0]");
  check(vx_topology_device_index(501, VX_TOPO_GPU_BASE) == 1, "GPU[1]");
  check(vx_topology_device_index(507, VX_TOPO_GPU_BASE) == 7, "GPU[7]");

  // Another kind's band is not this one's. Reading an NPU id as a GPU ordinal
  // would silently pick a device for hardware the launch was never meant for.
  check(vx_topology_device_index(100, VX_TOPO_GPU_BASE) == -1,
        "NPU is not GPU");
  check(vx_topology_device_index(0, VX_TOPO_GPU_BASE) == -1, "CPU is not GPU");
  check(vx_topology_device_index(600, VX_TOPO_GPU_BASE) == -1, "past the band");
  check(vx_topology_device_index(101, VX_TOPO_NPU_BASE) == 1,
        "NPU[1] in its own band");
}

void test_slot_decode() {
  SlotCase c(2, 3, 4);
  std::string payload =
      payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
  vx_gemm_plan plan;

  bool ok = vx_gemm_plan_decode(payload.data(), payload.size(), c.args.data(),
                                c.tags.data(), (int64_t)c.args.size(), &plan);
  check(ok, "the shape the compiler emits decodes");
  if (!ok) {
    return;
  }

  check(plan.m == 2 && plan.k == 3 && plan.n == 4, "extents from A and B");
  check(plan.dtype == VX_DTYPE_F32, "element type from the tags");
  check(plan.a_data == c.a_data.data(), "A is operand 0, not operand 2");
  check(plan.b_data == c.b_data.data(), "B is operand 2");
  check(plan.out_kind == VX_GEMM_OUT_SLOT, "outkind=slot");
  check(plan.out_desc == &c.result_desc,
        "the slot resolves to the descriptor storage, one indirection in");
  check(plan.out_data == nullptr, "a slot has no buffer to fill");

  // Publishing is the other half: the caller allocated a result and the
  // descriptor naming it has to land where the kernel would have stored one.
  std::vector<float> result((size_t)(plan.m * plan.n), 7.0f);
  vx_gemm_publish_slot(&plan, result.data());
  check(c.result_desc.aligned == result.data(), "published data pointer");
  check(c.result_desc.allocated == result.data(), "published allocation");
  check(c.result_desc.offset == 0, "published offset");
  check(c.result_desc.sizes[0] == 2 && c.result_desc.sizes[1] == 4,
        "published extents are M x N");
  check(c.result_desc.strides[0] == 4 && c.result_desc.strides[1] == 1,
        "published strides are row-major");
}

/// A tensor declared inside a function is captured as the slot it lives in,
/// so all three operands arrive one indirection further out than a function
/// parameter would. This is what the compiler emits for the ordinary case, and
/// it decoding correctly is the difference between a GEMM reaching the GPU and
/// every dispatch quietly falling back to the host.
void test_local_operands() {
  SlotCase c(2, 3, 4);
  c.inputs_as_locals();
  std::string payload =
      payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
  vx_gemm_plan plan;

  bool ok = vx_gemm_plan_decode(payload.data(), payload.size(), c.args.data(),
                                c.tags.data(), (int64_t)c.args.size(), &plan);
  check(ok, "operands reached through slots decode");
  if (!ok) {
    return;
  }

  check(plan.a_data == c.a_data.data() && plan.b_data == c.b_data.data(),
        "the data pointers are the buffers, not the slots");
  check(plan.m == 2 && plan.k == 3 && plan.n == 4,
        "extents read through the slots");
  check(plan.out_desc == &c.result_desc, "the result slot still resolves");
}

/// A result buffer reached through a slot is filled in place, not published:
/// `outkind` says which of the two the kernel does, and the tag says only how
/// the argument arrived. The two are independent.
void test_buffer_through_slot() {
  SlotCase c(2, 3, 4);
  c.inputs_as_locals();

  std::vector<float> out((size_t)(2 * 4), 0.0f);
  Desc2D out_desc = make_2d(out.data(), 2, 4, 4);
  SlotDesc out_slot = {&out_desc, &out_desc, 0};
  void *out_slot_ptr = &out_slot;
  c.args[5] = &out_slot_ptr;

  std::string payload =
      payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "buffer");
  vx_gemm_plan plan;

  bool ok = vx_gemm_plan_decode(payload.data(), payload.size(), c.args.data(),
                                c.tags.data(), (int64_t)c.args.size(), &plan);
  check(ok, "a buffer reached through a slot decodes");
  if (ok) {
    check(plan.out_kind == VX_GEMM_OUT_BUFFER, "still a buffer to fill");
    check(plan.out_data == out.data(), "the buffer the slot holds");
    check(plan.out_desc == nullptr, "nothing to publish");
  }
}

/// Roles are what make A and B distinguishable. With square operands the
/// shapes are identical, so a decode that fell back to argument order would
/// still look right here -- and would compute B*A.
void test_roles_beat_order() {
  SlotCase c(3, 3, 3);
  std::string payload =
      payload_of("vx_npu_kernel_0", "matmul", "a:2,b:0,out:5", "slot");
  vx_gemm_plan plan;

  bool ok = vx_gemm_plan_decode(payload.data(), payload.size(), c.args.data(),
                                c.tags.data(), (int64_t)c.args.size(), &plan);
  check(ok, "square operands decode");
  if (ok) {
    check(plan.a_data == c.b_data.data() && plan.b_data == c.a_data.data(),
          "the roles decide, not the operand order");
  }
}

void test_buffer_decode() {
  SlotCase c(2, 3, 4);
  std::vector<float> out((size_t)(2 * 4), 0.0f);
  Desc2D out_desc = make_2d(out.data(), 2, 4, 4);
  void *out_ptr = &out_desc;
  c.args[5] = &out_ptr;
  c.tags[5] = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);

  std::string payload =
      payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "buffer");
  vx_gemm_plan plan;

  bool ok = vx_gemm_plan_decode(payload.data(), payload.size(), c.args.data(),
                                c.tags.data(), (int64_t)c.args.size(), &plan);
  check(ok, "a captured result buffer decodes");
  if (ok) {
    check(plan.out_kind == VX_GEMM_OUT_BUFFER, "outkind=buffer");
    check(plan.out_data == out.data(), "the buffer to fill");
    check(plan.out_row_stride == 4, "its row stride");
    check(plan.out_desc == nullptr, "a buffer has no descriptor to publish");
  }

  // A result buffer of the wrong extent is not a GEMM this can stand in for.
  Desc2D wrong = make_2d(out.data(), 2, 3, 3);
  void *wrong_ptr = &wrong;
  c.args[5] = &wrong_ptr;
  check(!vx_gemm_plan_decode(payload.data(), payload.size(), c.args.data(),
                             c.tags.data(), (int64_t)c.args.size(), &plan),
        "result extents must match M x N");
}

/// Every refusal below leaves the caller running the outlined kernel, so being
/// strict costs performance and never correctness.
void test_refusals() {
  vx_gemm_plan plan;

  {
    SlotCase c(2, 3, 4);
    std::string p = payload_of("vx_npu_kernel_0", nullptr, nullptr, nullptr);
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "an unclassified kernel is not routed");
  }

  {
    // A payload predating the extension: size 0 means the entries were never
    // written, so the walk must report absence rather than read on.
    SlotCase c(2, 3, 4);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), 0, c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "a zero-sized payload is not walked");
  }

  {
    SlotCase c(2, 3, 4);
    std::string p =
        payload_of("vx_npu_kernel_0", "conv2d", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "only matmul is routed");
  }

  {
    SlotCase c(2, 3, 4);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:9", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "a role index past the argument list");
  }

  {
    SlotCase c(2, 3, 4);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:0,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "two roles naming one argument");
  }

  {
    // outkind=slot with a plain buffer tag, or the reverse: the payload and
    // the tags disagree about what the argument is, and writing a descriptor
    // over element data would corrupt the buffer silently.
    SlotCase c(2, 3, 4);
    c.tags[5] = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "outkind=slot needs the slot bit");
  }

  {
    // K disagrees between the operands, so no GEMM is defined.
    SlotCase c(2, 3, 4);
    c.b_desc.sizes[0] = 5;
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "inner dimensions must agree");
  }

  {
    // Mismatched element types: reading one operand as the other's type is
    // exactly the misinterpretation the tags were added to prevent.
    SlotCase c(2, 3, 4);
    c.tags[2] = VX_ABI_MEMREF_TAG(VX_DTYPE_F16, 2);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "operands must share an element type");
  }

  {
    SlotCase c(2, 3, 4);
    c.tags[0] = VX_ABI_MEMREF_TAG(VX_DTYPE_I32, 2);
    c.tags[2] = VX_ABI_MEMREF_TAG(VX_DTYPE_I32, 2);
    c.tags[5] = VX_ABI_SLOT_TAG(VX_DTYPE_I32, 2);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "an integer matmul runs as written");
  }

  {
    SlotCase c(2, 3, 4);
    c.tags[0] = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 3);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "operands must be rank 2");
  }

  {
    // Columns that are not adjacent need a transposed call, which the roles do
    // not describe.
    SlotCase c(2, 3, 4);
    c.a_desc.strides[1] = 2;
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                               (int64_t)c.args.size(), &plan),
          "rows must be contiguous");
  }

  {
    // Padded rows are fine: the stride is the leading dimension a GEMM takes.
    SlotCase c(2, 3, 4);
    c.a_desc.strides[0] = 8;
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                              (int64_t)c.args.size(), &plan) &&
              plan.a_row_stride == 8,
          "a padded row stride is carried, not rejected");
  }

  {
    // A view into a larger buffer starts partway in. Reading from the base
    // instead would be off by exactly the offset -- valid memory, wrong
    // numbers, which is the failure mode with no symptom.
    SlotCase c(2, 3, 4);
    c.a_desc.offset = 6;
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(vx_gemm_plan_decode(p.data(), p.size(), c.args.data(), c.tags.data(),
                              (int64_t)c.args.size(), &plan) &&
              plan.a_data == c.a_data.data() + 6,
          "a descriptor offset moves the data pointer");
  }
}

//===----------------------------------------------------------------------===//
// Attention plan decode (Vx#378): kind=attention, four rank-2 f16 memrefs
// named by q/k/v/out roles, the scale spelled `arg:<n>` or `val:<f>`.
//===----------------------------------------------------------------------===//

/// Four contiguous f16 operands the way `flash_attention_into` captures them
/// on the flat path: buffers directly, plus the captured f32 scale.
struct AttentionCase {
  std::vector<uint16_t> q_data, k_data, v_data, o_data;
  Desc2D q_desc, k_desc, v_desc, o_desc;
  float scale = 0.125f;

  void *q_ptr, *k_ptr, *v_ptr, *o_ptr;

  std::vector<void *> args;
  std::vector<int32_t> tags;

  AttentionCase(int64_t sq, int64_t sk, int64_t hd)
      : q_data((size_t)(sq * hd), 0), k_data((size_t)(sk * hd), 0),
        v_data((size_t)(sk * hd), 0), o_data((size_t)(sq * hd), 0) {
    q_desc = make_2d(q_data.data(), sq, hd, hd);
    k_desc = make_2d(k_data.data(), sk, hd, hd);
    v_desc = make_2d(v_data.data(), sk, hd, hd);
    o_desc = make_2d(o_data.data(), sq, hd, hd);

    q_ptr = &q_desc;
    k_ptr = &k_desc;
    v_ptr = &v_desc;
    o_ptr = &o_desc;

    args = {&o_ptr, &q_ptr, &k_ptr, &v_ptr, &scale};
    tags = {VX_ABI_MEMREF_TAG(VX_DTYPE_F16, 2),
            VX_ABI_MEMREF_TAG(VX_DTYPE_F16, 2),
            VX_ABI_MEMREF_TAG(VX_DTYPE_F16, 2),
            VX_ABI_MEMREF_TAG(VX_DTYPE_F16, 2), VX_ABI_KIND_F32};
  }
};

std::string payload_attention(const char *roles, const char *scale) {
  std::string p("vx_npu_kernel_0");
  p.push_back('\0');
  p += "kind=attention";
  p.push_back('\0');
  if (roles) {
    p += "roles=";
    p += roles;
    p.push_back('\0');
  }
  if (scale) {
    p += "scale=";
    p += scale;
    p.push_back('\0');
  }
  return p;
}

void test_parse_attention_roles() {
  int q = -1, k = -1, v = -1, o = -1;

  check(vx_parse_attention_roles("q:1,k:2,v:3,out:0", &q, &k, &v, &o) &&
            q == 1 && k == 2 && v == 3 && o == 0,
        "attention roles in the order the compiler emits them");

  q = k = v = o = -1;
  check(vx_parse_attention_roles("out:5,v:4,k:1,q:0", &q, &k, &v, &o) &&
            q == 0 && k == 1 && v == 4 && o == 5,
        "attention roles in any order");

  check(!vx_parse_attention_roles("q:1,k:2,v:3", &q, &k, &v, &o),
        "a missing attention role is a refusal");
  check(!vx_parse_attention_roles("q:1,k:2,v:3,out:0,a:4", &q, &k, &v, &o),
        "an unknown attention role");
  check(!vx_parse_attention_roles("q:1,q:2,v:3,out:0", &q, &k, &v, &o),
        "an attention role named twice");
  check(!vx_parse_attention_roles("q:1,k:2,v:3,out:-1", &q, &k, &v, &o),
        "a negative attention index");
  check(!vx_parse_attention_roles("", &q, &k, &v, &o), "empty attention roles");
  check(!vx_parse_attention_roles(nullptr, &q, &k, &v, &o),
        "absent attention roles");
}

void test_attention_decode() {
  vx_attention_plan plan;

  {
    AttentionCase c(8, 16, 64);
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "val:0.125");
    check(vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                   c.tags.data(), (int64_t)c.args.size(),
                                   &plan),
          "a noted attention region decodes");
    check(plan.sq == 8 && plan.sk == 16 && plan.hd == 64,
          "the dimensions come from the descriptors");
    check(plan.dtype == VX_DTYPE_F16, "the element type is f16");
    check(plan.q_data == c.q_data.data() && plan.k_data == c.k_data.data() &&
              plan.v_data == c.v_data.data() && plan.o_data == c.o_data.data(),
          "each role resolves to its own buffer");
    check(plan.scale == 0.125f, "a val: scale is the constant the region baked in");
  }

  {
    AttentionCase c(8, 16, 64);
    c.scale = 0.0883883f;
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "arg:4");
    check(vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                   c.tags.data(), (int64_t)c.args.size(),
                                   &plan) &&
              plan.scale == 0.0883883f,
          "an arg: scale reads the captured scalar at dispatch");
  }
}

void test_attention_refusals() {
  vx_attention_plan plan;

  {
    AttentionCase c(8, 16, 64);
    std::string p = payload_attention("q:1,k:2,v:3,out:0", nullptr);
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "attention without its scale is a different function");
  }

  {
    AttentionCase c(8, 16, 64);
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "0.125");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "a scale in neither spelling is a refusal, not a guess");
  }

  {
    AttentionCase c(8, 16, 64);
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "arg:9");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "a scale index past the argument list");
  }

  {
    AttentionCase c(8, 16, 64);
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "arg:0");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "a scale naming a memref is not an f32 scalar");
  }

  {
    AttentionCase c(8, 16, 64);
    std::string p = payload_attention("q:1,k:2,v:1,out:0", "val:0.125");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "two roles naming one operand");
  }

  {
    // f32 operands: correct math, but not the entry point that ships. The
    // refusal sends the region to its own kernel rather than to a wrong call.
    AttentionCase c(8, 16, 64);
    for (int i = 0; i < 4; ++i)
      c.tags[i] = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "val:0.125");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "only f16 attention has a vendor kernel today");
  }

  {
    // A padded row: the vendor entry point computes its own strides from the
    // dimensions, so padding would be read as data, silently.
    AttentionCase c(8, 16, 64);
    c.q_desc.strides[0] = 80;
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "val:0.125");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "a padded operand is not contiguous");
  }

  {
    // K and V must walk the same sequence; the descriptors say they do not.
    AttentionCase c(8, 16, 64);
    c.v_desc.sizes[0] = 12;
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "val:0.125");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "k and v of different lengths");
  }

  {
    AttentionCase c(8, 16, 64);
    c.o_desc.sizes[0] = 4;
    std::string p = payload_attention("q:1,k:2,v:3,out:0", "val:0.125");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "an output not shaped like the query");
  }

  {
    // A GEMM payload must not decode as attention, nor the reverse: the kinds
    // partition the routes.
    AttentionCase c(8, 16, 64);
    std::string p =
        payload_of("vx_npu_kernel_0", "matmul", "a:0,b:2,out:5", "slot");
    check(!vx_attention_plan_decode(p.data(), p.size(), c.args.data(),
                                    c.tags.data(), (int64_t)c.args.size(),
                                    &plan),
          "kind=matmul is not an attention");
  }
}

} // namespace

int main() {
  test_parse_roles();
  test_topology();
  test_slot_decode();
  test_local_operands();
  test_buffer_through_slot();
  test_roles_beat_order();
  test_buffer_decode();
  test_refusals();
  test_parse_attention_roles();
  test_attention_decode();
  test_attention_refusals();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
