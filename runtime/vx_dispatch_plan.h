//===- vx_dispatch_plan.h - Decode a dispatch into a GEMM plan --*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Turns the arguments of vx_plugin_dispatch_async into a description of the
// matrix multiply to run, or reports that there is none to run.
//
// This is deliberately free of any vendor API so that the decision -- which
// argument is A, which is B, where the result goes, what shape and type they
// are -- can be tested on a machine with no accelerator, which is where it will
// mostly be read and changed. runtime/cuda_dispatch.cpp consumes a plan and
// does the cuBLAS call; tests/runtime/gemm_plan_test.cpp builds descriptors by
// hand and checks the decode.
//
// Every field comes from something the compiler stated: `kind=matmul` that the
// region is a GEMM (VxLowering.cpp kernelKindOf), `roles=` which operand is
// which (matmulRolesOf), `outkind=` whether the result is a buffer to fill or a
// slot to publish through, and the per-argument tags the element type and rank.
// Nothing here is inferred from buffer shapes -- for square operands shapes
// cannot distinguish A from B, which is the reason #325 exists.
//
// Decoding fails, rather than substituting a default, whenever any of that does
// not line up. The caller's response to failure is to run the outlined kernel
// as written, so a refusal costs performance and never correctness.
//
//===----------------------------------------------------------------------===//

#ifndef VX_DISPATCH_PLAN_H
#define VX_DISPATCH_PLAN_H

#include "../include/vx_hardware_runtime.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Where a plugin must put the result it computed.
enum {
  /// The result buffer was handed to the kernel; fill it in place.
  VX_GEMM_OUT_BUFFER = 0,
  /// The kernel allocates its own result and stores the descriptor through a
  /// slot, so a plugin standing in for it must allocate and write one too.
  VX_GEMM_OUT_SLOT = 1
};

/// A row-major C[m,n] = A[m,k] * B[k,n], with the leading dimension of each
/// operand as it sits in memory. Rows are contiguous; `*_row_stride` is the
/// distance between them in elements and may exceed the row width.
typedef struct {
  int32_t dtype; /* VX_DTYPE_*, identical for all three operands */
  int64_t m, n, k;

  const void *a_data;
  const void *b_data;
  int64_t a_row_stride;
  int64_t b_row_stride;

  int32_t out_kind; /* VX_GEMM_OUT_BUFFER or VX_GEMM_OUT_SLOT */
  void *out_data;   /* buffer to fill; NULL when out_kind is a slot */
  int64_t out_row_stride;
  void *out_desc; /* descriptor storage to publish through; slot only */
} vx_gemm_plan;

/// Parse `a:<i>,b:<j>,out:<k>`. Returns 1 and fills the three indices, or 0 if
/// the string is not exactly that shape. A partial parse is a failure: filling
/// in the missing role by convention is the guessing this exists to replace.
static inline int vx_parse_roles(const char *roles, int *a, int *b, int *out) {
  const char *p = roles;
  int seen = 0;

  if (!roles || !a || !b || !out) {
    return 0;
  }

  while (*p) {
    int *target = NULL;
    int bit;
    long value;
    char *end;

    if (p[0] == 'a' && p[1] == ':') {
      target = a;
      bit = 1;
      p += 2;
    } else if (p[0] == 'b' && p[1] == ':') {
      target = b;
      bit = 2;
      p += 2;
    } else if (p[0] == 'o' && p[1] == 'u' && p[2] == 't' && p[3] == ':') {
      target = out;
      bit = 4;
      p += 4;
    } else {
      return 0;
    }

    /* A role named twice means the string is not the format this reads. */
    if (seen & bit) {
      return 0;
    }

    value = strtol(p, &end, 10);
    if (end == p || value < 0) {
      return 0;
    }
    *target = (int)value;
    seen |= bit;

    p = end;
    if (*p == ',') {
      /* A separator with nothing after it is malformed, not an end. */
      ++p;
      if (*p == '\0') {
        return 0;
      }
    } else if (*p != '\0') {
      return 0;
    }
  }

  return seen == 7;
}

/// True when a GEMM in this element type is worth routing to a vendor kernel.
/// Integer matmuls fall through to the outlined kernel: the vendor entry points
/// differ enough in signature that pretending otherwise would be the guess this
/// header avoids.
static inline int vx_gemm_dtype_supported(int32_t dtype) {
  return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_F64 ||
         dtype == VX_DTYPE_F16 || dtype == VX_DTYPE_BF16;
}

/// The descriptor an operand refers to.
///
/// A tensor declared inside a function lives in a slot -- storage holding a
/// descriptor -- and is captured as that slot, while a function parameter is
/// captured as the buffer itself. Both are ordinary operands to a matmul; the
/// tag says which arrived, so one dereference more or less is all that
/// separates them.
static inline const void *vx_operand_desc(void **device_args,
                                          const int32_t *arg_tags, int idx) {
  const void *desc = *(const void **)device_args[idx];
  if (desc && VX_ABI_IS_SLOT(arg_tags[idx])) {
    desc = (const void *)vx_memref_aligned(desc);
  }
  return desc;
}

/// Decode a dispatch into a plan. Returns 1 when the plan is safe to execute,
/// 0 when the caller should run the outlined kernel instead.
static inline int vx_gemm_plan_decode(const void *payload, size_t payload_size,
                                      void **device_args,
                                      const int32_t *arg_tags, int64_t num_args,
                                      vx_gemm_plan *plan) {
  const char *kind;
  const char *roles;
  const char *outkind;
  int ai = -1, bi = -1, oi = -1;
  int32_t a_tag, b_tag, o_tag;
  const void *a_desc;
  const void *b_desc;
  const void *o_desc;
  const int64_t *a_sizes;
  const int64_t *b_sizes;
  const int64_t *a_strides;
  const int64_t *b_strides;
  int32_t dtype;

  if (!plan || !device_args || !arg_tags) {
    return 0;
  }

  kind = vx_payload_field(payload, payload_size, "kind=");
  if (!kind || strcmp(kind, "matmul") != 0) {
    return 0;
  }

  roles = vx_payload_field(payload, payload_size, "roles=");
  if (!vx_parse_roles(roles, &ai, &bi, &oi)) {
    return 0;
  }
  if (ai >= num_args || bi >= num_args || oi >= num_args) {
    return 0;
  }
  if (ai == bi || ai == oi || bi == oi) {
    return 0;
  }

  outkind = vx_payload_field(payload, payload_size, "outkind=");
  if (!outkind) {
    return 0;
  }

  a_tag = arg_tags[ai];
  b_tag = arg_tags[bi];
  o_tag = arg_tags[oi];

  /* Rank-2 operands of one supported float type. Each may have arrived as a
     buffer or as the slot holding one; the tag says which, and the element type
     and rank describe the buffer either way. */
  if (VX_ABI_KIND(a_tag) != VX_ABI_KIND_MEMREF ||
      VX_ABI_KIND(b_tag) != VX_ABI_KIND_MEMREF ||
      VX_ABI_KIND(o_tag) != VX_ABI_KIND_MEMREF) {
    return 0;
  }
  if (VX_ABI_RANK(a_tag) != 2 || VX_ABI_RANK(b_tag) != 2 ||
      VX_ABI_RANK(o_tag) != 2) {
    return 0;
  }

  dtype = VX_ABI_ELEM(a_tag);
  if (!vx_gemm_dtype_supported(dtype) || VX_ABI_ELEM(b_tag) != dtype ||
      VX_ABI_ELEM(o_tag) != dtype) {
    return 0;
  }

  a_desc = vx_operand_desc(device_args, arg_tags, ai);
  b_desc = vx_operand_desc(device_args, arg_tags, bi);
  o_desc = vx_operand_desc(device_args, arg_tags, oi);
  if (!a_desc || !b_desc || !o_desc) {
    return 0;
  }

  a_sizes = vx_memref_sizes(a_desc);
  b_sizes = vx_memref_sizes(b_desc);
  a_strides = vx_memref_strides(a_desc, 2);
  b_strides = vx_memref_strides(b_desc, 2);

  /* Row-major with contiguous rows. A column-major or otherwise permuted
     operand would need a transposed call, which the roles do not describe. */
  if (a_strides[1] != 1 || b_strides[1] != 1) {
    return 0;
  }
  if (a_sizes[0] <= 0 || a_sizes[1] <= 0 || b_sizes[1] <= 0) {
    return 0;
  }
  if (b_sizes[0] != a_sizes[1]) {
    return 0;
  }

  plan->dtype = dtype;
  plan->m = a_sizes[0];
  plan->k = a_sizes[1];
  plan->n = b_sizes[1];
  /* The offset is part of where the elements are: a view into a larger buffer
     starts partway in, and reading from the base would silently be off. */
  plan->a_data = vx_memref_data(a_desc, dtype);
  plan->b_data = vx_memref_data(b_desc, dtype);
  plan->a_row_stride = a_strides[0];
  plan->b_row_stride = b_strides[0];
  plan->out_data = NULL;
  plan->out_desc = NULL;
  plan->out_row_stride = plan->n;

  if (strcmp(outkind, "slot") == 0) {
    /* Publishing means writing a descriptor, so there has to be storage to
       write it into -- which is what the slot bit says the argument is. */
    if (!VX_ABI_IS_SLOT(o_tag)) {
      return 0;
    }
    plan->out_kind = VX_GEMM_OUT_SLOT;
    plan->out_desc = (void *)o_desc;
  } else if (strcmp(outkind, "buffer") == 0) {
    /* Filling a buffer in place. Whether it was captured directly or reached
       through the slot a local tensor lives in, o_desc describes the buffer. */
    const int64_t *o_sizes = vx_memref_sizes(o_desc);
    const int64_t *o_strides = vx_memref_strides(o_desc, 2);
    if (o_strides[1] != 1) {
      return 0;
    }
    if (o_sizes[0] != plan->m || o_sizes[1] != plan->n) {
      return 0;
    }
    plan->out_kind = VX_GEMM_OUT_BUFFER;
    plan->out_data = vx_memref_data(o_desc, dtype);
    plan->out_row_stride = o_strides[0];
    if (!plan->out_data) {
      return 0;
    }
  } else {
    return 0;
  }

  if (!plan->a_data || !plan->b_data) {
    return 0;
  }

  return 1;
}

/// Publish a result computed for a slot: `data` is the freshly allocated result
/// buffer, and the descriptor naming it is written where the kernel would have
/// stored its own.
static inline void vx_gemm_publish_slot(const vx_gemm_plan *plan, void *data) {
  int64_t sizes[2];
  sizes[0] = plan->m;
  sizes[1] = plan->n;
  vx_memref_write_desc(plan->out_desc, data, 2, sizes);
}

#ifdef __cplusplus
}
#endif

#endif /* VX_DISPATCH_PLAN_H */
