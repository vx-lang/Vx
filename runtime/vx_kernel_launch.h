//===- vx_kernel_launch.h - Marshalling a dispatch to a kernel --*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// A dispatch names a kernel and carries its arguments in the plugin ABI's
// shape: one pointer per C-interface argument, plus a tag saying what each one
// is. A device entry point wants something else -- one parameter per *field* of
// each memref -- and this turns the first into the second (#251).
//
// The two shapes are closer than they look. `convert-gpu-to-nvvm` explodes a
// `memref<32x16xf32>` parameter into seven `.u64`s: allocated pointer, aligned
// pointer, offset, one size per dimension, one stride per dimension. A
// descriptor in memory is *already* exactly that, in exactly that order, with
// every field eight bytes wide -- which is why nothing here copies. The
// parameter list is addresses into descriptors the caller already has, which is
// also the form `cuLaunchKernel` wants (it takes pointers to values, not the
// values).
//
// So the whole marshalling is an arithmetic claim about a layout, and getting
// it wrong launches a kernel that reads its arguments from the wrong offsets
// and writes somewhere adjacent. That is why this is a vendor-free header with
// a test (tests/runtime/kernel_launch_test.cpp) rather than a loop inside
// runtime/cuda_dispatch.cpp: it can then be checked on a laptop instead of
// discovered on a rented GPU.
//
//===----------------------------------------------------------------------===//

#ifndef VX_KERNEL_LAUNCH_H
#define VX_KERNEL_LAUNCH_H

#include "../include/vx_hardware_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/// Enough for eight rank-3 memrefs, or the four rank-2 ones the attention
/// kernel takes with room to spare. A kernel wanting more than this is not a
/// kernel this path should be quietly truncating for.
#define VX_LAUNCH_MAX_PARAMS 128

typedef struct {
  void *params[VX_LAUNCH_MAX_PARAMS];
  int count;
} vx_launch_params;

/// How many device parameters an argument of this tag expands into.
///
/// Returns 0 for a tag this path cannot marshal, which is the caller's cue to
/// refuse rather than to guess a width.
static inline int vx_launch_param_width(int32_t tag) {
  /* A slot is storage the kernel publishes a result *into*, by writing a
     descriptor it allocated. Its device-side form is a pointer to a descriptor
     that does not exist yet, so there is nothing here to take the address of.
     That is the unresolved half of #251 and it is refused, not approximated. */
  if (VX_ABI_IS_SLOT(tag)) {
    return 0;
  }
  if (VX_ABI_KIND(tag) == VX_ABI_KIND_MEMREF) {
    int32_t rank = (int32_t)VX_ABI_RANK(tag);
    if (rank < 0 || rank > VX_ABI_MAX_RANK) {
      return 0;
    }
    /* allocated, aligned, offset, then a size and a stride per dimension. */
    return 3 + 2 * rank;
  }
  /* A scalar is one parameter, passed by address like everything else. */
  return vx_dtype_bytes(VX_ABI_KIND(tag)) ? 1 : 0;
}

/// Flatten a dispatch's arguments into a device parameter list.
///
/// `device_args[i]` is what the plugin ABI hands over: for a memref it points
/// at a pointer to the descriptor, for a scalar it points at the value. So a
/// memref costs one dereference to reach the descriptor, and the parameters are
/// then addresses of its fields, in declaration order.
///
/// Returns 0 without having produced a partial list when an argument cannot be
/// marshalled or the list would not fit. A partial list is the dangerous
/// outcome: it launches, and the kernel reads whatever the remaining parameter
/// slots happened to contain.
static inline int vx_launch_build_params(void *const *device_args,
                                         const int32_t *arg_tags,
                                         int64_t num_args,
                                         vx_launch_params *out) {
  int count = 0;
  int64_t i;

  if (!device_args || !arg_tags || !out || num_args < 0) {
    return 0;
  }
  out->count = 0;

  for (i = 0; i < num_args; ++i) {
    const int32_t tag = arg_tags[i];
    const int width = vx_launch_param_width(tag);
    if (width == 0 || count + width > VX_LAUNCH_MAX_PARAMS) {
      return 0;
    }
    if (!device_args[i]) {
      return 0;
    }

    if (VX_ABI_KIND(tag) != VX_ABI_KIND_MEMREF) {
      out->params[count++] = (void *)device_args[i];
      continue;
    }

    {
      /* The descriptor itself, not the argument slot pointing at it. */
      char *desc = (char *)*(void *const *)device_args[i];
      const int32_t rank = (int32_t)VX_ABI_RANK(tag);
      int32_t d;
      if (!desc) {
        return 0;
      }
      /* {allocated, aligned} -- two pointers, in that order. */
      out->params[count++] = desc;
      out->params[count++] = desc + sizeof(void *);
      /* offset */
      out->params[count++] = desc + 2 * sizeof(void *);
      /* sizes, then strides: contiguous int64s following the offset, which is
         what vx_memref_sizes/vx_memref_strides read and what
         vx_memref_write_desc writes. */
      for (d = 0; d < rank; ++d) {
        out->params[count++] =
            (void *)((const char *)vx_memref_sizes(desc) + d * sizeof(int64_t));
      }
      for (d = 0; d < rank; ++d) {
        out->params[count++] =
            (void *)((const char *)vx_memref_strides(desc, rank) +
                     d * sizeof(int64_t));
      }
    }
  }

  out->count = count;
  return 1;
}

/// How many parameters an entry point in this PTX declares.
///
/// Returns -1 when the entry is not found, so "absent" and "takes none" stay
/// distinguishable.
///
/// This exists to be compared against vx_launch_build_params' count before
/// anything is launched. A mismatch means the compiler's idea of the kernel's
/// signature and the runtime's have diverged, and the failure mode without the
/// check is silent: `cuLaunchKernel` is handed an array it cannot size-check,
/// so the kernel reads parameters past the ones supplied and computes from
/// whatever is there. The image is right here in the payload and the count is a
/// text scan, so there is no reason to launch on faith.
///
/// Counting is confined to the parenthesised signature. Counting every
/// `.param` in the file also counts each `ld.param` and every extern's return
/// slot, which turned 28 into 36 when scripts/flash_kernel_to_ptx.sh first
/// tried it.
static inline int vx_launch_entry_param_count(const char *ptx,
                                              const char *entry) {
  const char *p = ptx;
  size_t entry_len;

  if (!ptx || !entry) {
    return -1;
  }
  entry_len = strlen(entry);

  for (;;) {
    const char *open;
    const char *close;
    int count;

    p = strstr(p, ".entry ");
    if (!p) {
      return -1;
    }
    p += 7;
    while (*p == ' ' || *p == '\t') {
      ++p;
    }
    if (strncmp(p, entry, entry_len) != 0) {
      continue;
    }
    /* A prefix match is not a match: `vx_npu_kernel_1` must not answer for
       `vx_npu_kernel_10`, and both exist as soon as a program has eleven
       regions. */
    if (p[entry_len] != '(' && p[entry_len] != ' ' && p[entry_len] != '\t' &&
        p[entry_len] != '\n') {
      continue;
    }

    open = strchr(p, '(');
    if (!open) {
      return -1;
    }
    close = strchr(open, ')');
    if (!close) {
      return -1;
    }

    count = 0;
    for (p = open; p < close; ++p) {
      if (strncmp(p, ".param", 6) == 0) {
        ++count;
      }
    }
    return count;
  }
}

#endif /* VX_KERNEL_LAUNCH_H */
