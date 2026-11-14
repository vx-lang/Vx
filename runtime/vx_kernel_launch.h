//===- vx_kernel_launch.h - Marshalling a dispatch to a kernel --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// Enough for eight rank-3 memrefs, or the four rank-2 ones a four-operand
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
/// slot, which turned 28 into 36 when the emitted PTX was first
/// tried it.
/// The signature of an entry point in this PTX: `*open` at its `(` and
/// `*close` at its `)`. Returns 0 when the entry is not found.
static inline int vx_launch_entry_signature(const char *ptx, const char *entry,
                                            const char **open,
                                            const char **close) {
  const char *p = ptx;
  size_t entry_len;

  if (!ptx || !entry) {
    return 0;
  }
  entry_len = strlen(entry);

  for (;;) {
    p = strstr(p, ".entry ");
    if (!p) {
      return 0;
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

    *open = strchr(p, '(');
    if (!*open) {
      return 0;
    }
    *close = strchr(*open, ')');
    return *close != NULL;
  }
}

static inline int vx_launch_entry_param_count(const char *ptx,
                                              const char *entry) {
  const char *open;
  const char *close;
  const char *p;
  int count = 0;

  if (!vx_launch_entry_signature(ptx, entry, &open, &close)) {
    return -1;
  }
  for (p = open; p < close; ++p) {
    if (strncmp(p, ".param", 6) == 0) {
      ++count;
    }
  }
  return count;
}

/// The alignment that each parameter of an entry point says it has, from its
/// `.ptr .align N`, into `align[0..max)`: 0 for a parameter that is not such a
/// pointer. Returns 0 when the entry is not found.
///
/// The compiler puts `.align N` on the pointer of a buffer whose memory space
/// has `alignment: N` in the machine file. The kernel can load it with vector
/// loads of that width, and a pointer with less alignment faults there.
static inline int vx_launch_param_aligns(const char *ptx, const char *entry,
                                         long *align, int max) {
  const char *open;
  const char *close;
  const char *p;
  int k = -1;

  if (!align || !vx_launch_entry_signature(ptx, entry, &open, &close)) {
    return 0;
  }
  for (p = open; p < close; ++p) {
    const char *end;
    const char *at;
    const char *ptr;
    if (strncmp(p, ".param", 6) != 0) {
      continue;
    }
    if (++k >= max) {
      break;
    }
    align[k] = 0;
    end = strchr(p + 6, ',');
    if (!end || end > close) {
      end = close;
    }
    ptr = strstr(p, ".ptr");
    at = strstr(p, ".align ");
    if (ptr && ptr < end && at && at < end) {
      align[k] = strtol(at + 7, NULL, 10);
    }
  }
  return 1;
}

/// The first parameter whose pointer has less alignment than `align` says, or
/// -1 when there is none. A view over memory from elsewhere can start at any
/// address, so this is checked before each launch.
static inline int vx_launch_misaligned(const vx_launch_params *params,
                                       const long *align, int count) {
  int k;
  for (k = 0; k < count && k < params->count; ++k) {
    uintptr_t v;
    if (align[k] <= 1) {
      continue;
    }
    memcpy(&v, params->params[k], sizeof v);
    if (v % (uintptr_t)align[k] != 0) {
      return k;
    }
  }
  return -1;
}

/* One size of a launch shape, from `text`: `$j` is the `i32` launch value j,
   which the compiled code puts after the kernel's `num_args` arguments, and a
   number is that size. Moves `*text` past the entry and its comma. Returns -1
   when launch value j is not an `i32`. */
static inline long vx_launch_size(const char **text, void *const *device_args,
                                  const int32_t *arg_tags, int64_t num_args) {
  char *end = NULL;
  long value;
  if (**text == '$') {
    long j = strtol(*text + 1, &end, 10);
    /* A shape has at most four sizes. */
    if (j < 0 || j > 3 || arg_tags[num_args + j] != 4) {
      return -1;
    }
    value = *(const int32_t *)device_args[num_args + j];
  } else {
    value = strtol(*text, &end, 10);
  }
  *text = end;
  if (**text == ',') {
    ++*text;
  }
  return value;
}

/* The launch shape of a region, from the payload's `grid=` and `block=`
   entries, which the `across` loops give (#252). Returns 1 and fills `grid` and
   `block`, 0 when the payload has no shape, or -1 with the reason in `why` when
   a size is not one a GPU can launch. A smaller launch would skip work, so the
   caller stops the program on -1. With `grid_loops=1`, a loop of blocks steps
   over the blocks of the grid, so any grid covers its passes: a grid size below
   1 is 1, and one above what the GPU launches on that axis is that limit. */
static inline int vx_launch_shape(const void *payload, size_t payload_size,
                                  void *const *device_args,
                                  const int32_t *arg_tags, int64_t num_args,
                                  unsigned grid[3], unsigned *block, char *why,
                                  size_t why_size) {
  static const long limits[3] = {2147483647L, 65535L, 65535L};
  const char *at = vx_payload_field(payload, payload_size, "grid=");
  const char *block_text = vx_payload_field(payload, payload_size, "block=");
  if (!at || !block_text) {
    return 0;
  }
  const int loops =
      vx_payload_field(payload, payload_size, "grid_loops=") != NULL;
  for (int axis = 0; axis < 3; ++axis) {
    long n = *at ? vx_launch_size(&at, device_args, arg_tags, num_args) : 1;
    if (loops) {
      n = n < 1 ? 1 : n > limits[axis] ? limits[axis] : n;
    } else if (n < 1 || n > limits[axis]) {
      snprintf(why, why_size,
               "grid axis %d has %ld blocks, and a GPU launches 1 to %ld on "
               "that axis",
               axis, n, limits[axis]);
      return -1;
    }
    grid[axis] = (unsigned)n;
  }
  long threads = vx_launch_size(&block_text, device_args, arg_tags, num_args);
  if (threads < 1 || threads > 1024) {
    snprintf(why, why_size,
             "a block of %ld threads, and a block holds 1 to 1024 threads",
             threads);
    return -1;
  }
  *block = (unsigned)threads;
  return 1;
}

#endif /* VX_KERNEL_LAUNCH_H */
