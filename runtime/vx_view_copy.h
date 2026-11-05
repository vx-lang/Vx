//===- vx_view_copy.h - Copy the views of one call --------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A plugin that runs a region away from the memory of its tensors copies each
// view there for the call, and copies it back after the call. The CUDA plugin
// copies device views to the host, and the remote client copies host views to
// a worker.
//
// Two views can share memory, for example one tensor passed as the output and
// as the input. If each view has its own copy, the copy back of one view puts
// old values over the result in the other. So views whose bytes overlap form
// one span, and each span is copied once in each direction.
//
// A span holds every byte from a view's first element to its last, so one
// column of a matrix spans almost the whole matrix. A view that has a span to
// itself gets a dense copy instead: only its elements, with dense strides
// during the call. A view that shares its span keeps its strides, because the
// views must see one copy.
//
// This header plans the spans and needs no vendor library, so
// tests/runtime/view_copy_test.cpp runs it on any machine.
//
//===----------------------------------------------------------------------===//

#ifndef VX_VIEW_COPY_H
#define VX_VIEW_COPY_H

#include "../include/vx_hardware_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// One view of a call.
typedef struct {
  /* The caller fills these. */
  void *desc; /* the view's descriptor */
  int32_t rank;
  int32_t elem; /* the element type code */

  /* vx_view_copy_plan fills these. */
  uintptr_t start; /* the address of the view's first element */
  uint64_t bytes;  /* the bytes the view reaches, 0 for an empty view */
  int64_t span;    /* the span that holds the view, -1 for an empty view */
  uint64_t at;     /* the byte where the view starts in its span */
  int32_t packed;  /* 1 when the view has a dense copy of its own */
  int64_t sizes[VX_ABI_MAX_RANK]; /* the view's layout, when it is packed */
  int64_t strides[VX_ABI_MAX_RANK];

  /* vx_view_copy_point_at fills these, and vx_view_copy_restore reads them. */
  void *allocated;
  void *aligned;
  int64_t offset;
} vx_view_copy;

/// Bytes that one or more views reach, copied as one block.
typedef struct {
  uintptr_t start; /* the address of the span's first byte */
  uint64_t bytes;  /* the bytes of the copy */
  void *copy;      /* the caller's copy of the span */
  int64_t view;    /* the view of a dense copy, or -1 for a copy of the bytes */
} vx_copy_span;

/// The root of the group that holds view `i`. The group links run from a view
/// to a view with a smaller index, so the root is the group's first view.
static inline int64_t vx_view_copy_root(const vx_view_copy *views, int64_t i) {
  while (views[i].span != i) {
    i = views[i].span;
  }
  return i;
}

/// Whether the view `v` gets a dense copy, and its bytes in `*dense`: when its
/// elements need fewer bytes than the span from its first element to its
/// last, each stride is above 0, and it reaches no element twice.
static inline int vx_view_copy_packable(vx_view_copy *v, uint64_t *dense) {
  const int64_t *sizes = vx_memref_sizes(v->desc);
  const int64_t *strides = vx_memref_strides(v->desc, v->rank);
  int32_t dims[VX_ABI_MAX_RANK], n = 0;
  uint64_t elements = 1;
  if (v->rank > VX_ABI_MAX_RANK) {
    return 0;
  }
  for (int32_t d = 0; d < v->rank; ++d) {
    v->sizes[d] = sizes[d];
    v->strides[d] = strides[d];
    elements *= (uint64_t)sizes[d];
    if (sizes[d] > 1) {
      if (strides[d] <= 0) {
        return 0;
      }
      dims[n++] = d;
    }
  }
  /* Sorted by stride, each dimension must start past the end of the one
     before it. */
  for (int32_t i = 1; i < n; ++i) {
    for (int32_t j = i; j > 0 && strides[dims[j]] < strides[dims[j - 1]]; --j) {
      int32_t t = dims[j];
      dims[j] = dims[j - 1];
      dims[j - 1] = t;
    }
  }
  for (int32_t i = 1; i < n; ++i) {
    if (strides[dims[i]] < strides[dims[i - 1]] * sizes[dims[i - 1]]) {
      return 0;
    }
  }
  *dense = elements * vx_dtype_bytes(v->elem);
  return *dense < v->bytes;
}

/// Plan the copies of `n` views. Fills the plan fields of each view, and
/// writes the spans to `spans`, which has room for `n`.
///
/// Returns 1. Returns 0 and sets `*bad` to the view's index when a view has a
/// negative offset, or when its bytes do not fit in the address space.
/// Stops the program for a broken view, as vx_memref_view_bytes does.
static inline int vx_view_copy_plan(vx_view_copy *views, int64_t n,
                                    vx_copy_span *spans, int64_t *num_spans,
                                    int64_t *bad) {
  *num_spans = 0;
  for (int64_t i = 0; i < n; ++i) {
    vx_view_copy *v = &views[i];
    const int64_t *sizes = vx_memref_sizes(v->desc);
    const int64_t offset = vx_memref_offset(v->desc);
    uint64_t first, end;

    v->bytes = vx_memref_view_bytes(sizes, vx_memref_strides(v->desc, v->rank),
                                    v->rank, v->elem);
    v->span = -1;
    v->start = 0;
    v->at = 0;
    if (v->bytes == 0) {
      continue;
    }
    if (offset < 0 ||
        __builtin_mul_overflow((uint64_t)offset,
                               (uint64_t)vx_dtype_bytes(v->elem), &first) ||
        __builtin_add_overflow((uint64_t)(uintptr_t)vx_memref_aligned(v->desc),
                               first, &first) ||
        __builtin_add_overflow(first, v->bytes, &end) ||
        end > (uint64_t)UINTPTR_MAX) {
      *bad = i;
      return 0;
    }
    v->start = (uintptr_t)first;
    v->span = i;
  }

  /* Join each two views whose bytes overlap. A group joined this way covers
     one block with no gaps, so a view that overlaps the group overlaps one of
     its views. */
  for (int64_t i = 0; i < n; ++i) {
    if (views[i].span < 0) {
      continue;
    }
    for (int64_t j = 0; j < i; ++j) {
      int64_t a, b;
      if (views[j].span < 0 ||
          views[i].start >= views[j].start + views[j].bytes ||
          views[j].start >= views[i].start + views[i].bytes) {
        continue;
      }
      a = vx_view_copy_root(views, i);
      b = vx_view_copy_root(views, j);
      if (a < b) {
        views[b].span = a;
      } else if (b < a) {
        views[a].span = b;
      }
    }
  }

  /* Give each group a span. A root has the smallest index in its group, so it
     gets its span before the other views in the group. `at` holds the root
     until the last loop. */
  for (int64_t i = 0; i < n; ++i) {
    if (views[i].span >= 0) {
      views[i].at = (uint64_t)vx_view_copy_root(views, i);
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    vx_view_copy *v = &views[i];
    vx_copy_span *s;
    uint64_t end;
    if (v->span < 0) {
      continue;
    }
    if (v->at == (uint64_t)i) {
      s = &spans[*num_spans];
      s->start = v->start;
      s->bytes = v->bytes;
      s->copy = NULL;
      v->span = (*num_spans)++;
      continue;
    }
    v->span = views[v->at].span;
    s = &spans[v->span];
    end = s->start + s->bytes;
    if (v->start + v->bytes > end) {
      end = v->start + v->bytes;
    }
    if (v->start < s->start) {
      s->start = v->start;
    }
    s->bytes = end - s->start;
  }
  for (int64_t i = 0; i < n; ++i) {
    views[i].at =
        views[i].span < 0 ? 0 : views[i].start - spans[views[i].span].start;
    views[i].packed = 0;
  }

  /* A span with one view gets a dense copy when that is smaller. */
  for (int64_t s = 0; s < *num_spans; ++s) {
    int64_t only = -1, count = 0;
    uint64_t dense = 0;
    spans[s].view = -1;
    for (int64_t i = 0; i < n; ++i) {
      if (views[i].span == s) {
        only = i;
        ++count;
      }
    }
    if (count == 1 && vx_view_copy_packable(&views[only], &dense)) {
      views[only].packed = 1;
      spans[s].bytes = dense;
      spans[s].view = only;
    }
  }
  return 1;
}

/// Calls `block(ctx, dense, view, width, pitch, height)` for each block of the
/// packed view `v`: `height` runs of `width` bytes, `pitch` bytes apart in the
/// view, and back to back from byte `dense` of the dense copy. The blocks come
/// in the order of the dense copy.
typedef void (*vx_view_copy_block)(void *ctx, uint64_t dense, uintptr_t view,
                                   uint64_t width, uint64_t pitch,
                                   uint64_t height);

static inline void vx_view_copy_blocks(const vx_view_copy *v,
                                       vx_view_copy_block block, void *ctx) {
  const uint64_t b = vx_dtype_bytes(v->elem);
  int32_t dims[VX_ABI_MAX_RANK]; /* the dimensions of more than 1 element */
  int64_t idx[VX_ABI_MAX_RANK] = {0};
  int32_t n = 0, row, d;
  uint64_t width, pitch, height, dense = 0;
  for (d = 0; d < v->rank; ++d) {
    if (v->sizes[d] > 1) {
      dims[n++] = d;
    }
  }
  if (n == 0) {
    block(ctx, 0, v->start, b, b, 1);
    return;
  }
  /* A last dimension with a stride of 1 is one run. */
  if (v->strides[dims[n - 1]] == 1) {
    width = (uint64_t)v->sizes[dims[n - 1]] * b;
    row = n - 2;
  } else {
    width = b;
    row = n - 1;
  }
  if (row < 0) {
    block(ctx, 0, v->start, width, width, 1);
    return;
  }
  height = (uint64_t)v->sizes[dims[row]];
  pitch = (uint64_t)v->strides[dims[row]] * b;
  for (;;) {
    uint64_t at = 0;
    for (d = 0; d < row; ++d) {
      at += (uint64_t)idx[d] * (uint64_t)v->strides[dims[d]] * b;
    }
    block(ctx, dense, v->start + (uintptr_t)at, width, pitch, height);
    dense += width * height;
    /* The next block: count up the dimensions before the row, last first. */
    for (d = row - 1; d >= 0; --d) {
      if (++idx[d] < v->sizes[dims[d]]) {
        break;
      }
      idx[d] = 0;
    }
    if (d < 0) {
      return;
    }
  }
}

static inline void vx_view_copy_gather_block(void *ctx, uint64_t dense,
                                             uintptr_t view, uint64_t width,
                                             uint64_t pitch, uint64_t height) {
  for (uint64_t h = 0; h < height; ++h) {
    memcpy((char *)ctx + dense + h * width, (const char *)view + h * pitch,
           width);
  }
}

static inline void vx_view_copy_scatter_block(void *ctx, uint64_t dense,
                                              uintptr_t view, uint64_t width,
                                              uint64_t pitch, uint64_t height) {
  for (uint64_t h = 0; h < height; ++h) {
    memcpy((char *)view + h * pitch, (const char *)ctx + dense + h * width,
           width);
  }
}

/// Copy the elements of the packed view `v`, in local memory, into `dense`.
static inline void vx_view_copy_gather(const vx_view_copy *v, void *dense) {
  vx_view_copy_blocks(v, vx_view_copy_gather_block, dense);
}

/// Copy `dense` back into the elements of the packed view `v`.
static inline void vx_view_copy_scatter(const vx_view_copy *v, void *dense) {
  vx_view_copy_blocks(v, vx_view_copy_scatter_block, dense);
}

/// The stride of dimension `d` in the dense copy of `v`.
static inline int64_t vx_view_copy_dense_stride(const vx_view_copy *v,
                                                int32_t d) {
  int64_t stride = 1;
  for (int32_t k = v->rank - 1; k > d; --k) {
    stride *= v->sizes[k];
  }
  return stride;
}

/// Point each view's descriptor at its place in the copy of its span. The
/// offset becomes 0, because a copy holds no elements before its span. An
/// empty view keeps its descriptor, because a region reads none of its
/// elements.
static inline void vx_view_copy_point_at(vx_view_copy *views, int64_t n,
                                         const vx_copy_span *spans) {
  for (int64_t i = 0; i < n; ++i) {
    vx_view_copy *v = &views[i];
    void **ptrs = (void **)v->desc;
    void *at;
    if (v->span < 0) {
      continue;
    }
    at = (char *)spans[v->span].copy + v->at;
    v->allocated = ptrs[0];
    v->aligned = ptrs[1];
    v->offset = vx_memref_offset(v->desc);
    ptrs[0] = at;
    ptrs[1] = at;
    *(int64_t *)((char *)v->desc + 2 * sizeof(void *)) = 0;
    if (v->packed) {
      int64_t *strides = (int64_t *)vx_memref_strides(v->desc, v->rank);
      for (int32_t d = 0; d < v->rank; ++d) {
        strides[d] = vx_view_copy_dense_stride(v, d);
      }
    }
  }
}

/// A descriptor of rank `rank` that points into the dense copy of `v`, with
/// the offset `*offset` and the strides `strides` in the dense layout: the same
/// elements in the layout of `v`. Returns 0 when that is not exact: a stride
/// is not a multiple of one dense stride, or the descriptor reaches past the
/// copy.
static inline int vx_view_copy_unpack(const vx_view_copy *v, int64_t *offset,
                                      const int64_t *sizes, int64_t *strides,
                                      int32_t rank) {
  int64_t idx[VX_ABI_MAX_RANK], reach[VX_ABI_MAX_RANK] = {0};
  int64_t rest = *offset, out = v->offset;
  int32_t into[VX_ABI_MAX_RANK]; /* the dimension of `v` of each stride */
  int64_t times[VX_ABI_MAX_RANK];
  if (rest < 0 || rank > VX_ABI_MAX_RANK) {
    return 0;
  }
  for (int32_t k = 0; k < v->rank; ++k) {
    idx[k] = rest / vx_view_copy_dense_stride(v, k);
    rest -= idx[k] * vx_view_copy_dense_stride(v, k);
  }
  if (rest != 0) {
    return 0;
  }
  for (int32_t j = 0; j < rank; ++j) {
    into[j] = -1;
    times[j] = 0;
    if (sizes[j] <= 1) {
      continue;
    }
    /* Dense strides of dimensions of more than 1 element only go down, so the
       first one that divides the stride is the largest. */
    for (int32_t k = 0; k < v->rank && into[j] < 0; ++k) {
      int64_t dense = vx_view_copy_dense_stride(v, k);
      if (v->sizes[k] > 1 && strides[j] > 0 && strides[j] % dense == 0) {
        into[j] = k;
        times[j] = strides[j] / dense;
        reach[k] += (sizes[j] - 1) * times[j];
      }
    }
    if (into[j] < 0) {
      return 0;
    }
  }
  for (int32_t k = 0; k < v->rank; ++k) {
    if (idx[k] + reach[k] >= (v->sizes[k] > 0 ? v->sizes[k] : 1)) {
      return 0;
    }
    out += idx[k] * v->strides[k];
  }
  for (int32_t j = 0; j < rank; ++j) {
    strides[j] = into[j] < 0 ? 0 : times[j] * v->strides[into[j]];
  }
  *offset = out;
  return 1;
}

/// Undo vx_view_copy_point_at in every descriptor in `descs` that still
/// points at a view's copy. That includes the views' own descriptors, and a
/// descriptor that the region published from a view. A published descriptor
/// can have its own offset, so the view's old offset is added to it. In a
/// dense copy, its offset and strides go back to the view's layout. `ranks`
/// gives the rank of each descriptor.
///
/// Stops the program when a descriptor points into a dense copy in a way that
/// has no exact layout in the view, so that no wrong view is left.
static inline void vx_view_copy_restore(const vx_view_copy *views, int64_t n,
                                        const vx_copy_span *spans,
                                        void *const *descs,
                                        const int32_t *ranks,
                                        int64_t num_descs) {
  for (int64_t i = 0; i < n; ++i) {
    const vx_view_copy *v = &views[i];
    void *at;
    if (v->span < 0) {
      continue;
    }
    at = (char *)spans[v->span].copy + v->at;
    for (int64_t k = 0; k < num_descs; ++k) {
      void **d = (void **)descs[k];
      int64_t *offset;
      if (!d || d[1] != at) {
        continue;
      }
      d[0] = v->allocated;
      d[1] = v->aligned;
      offset = (int64_t *)((char *)d + 2 * sizeof(void *));
      if (!v->packed) {
        *offset += v->offset;
      } else if ((void *)d == v->desc && *offset == 0) {
        int64_t *strides = (int64_t *)vx_memref_strides(d, v->rank);
        *offset = v->offset;
        for (int32_t r = 0; r < v->rank; ++r) {
          strides[r] = v->strides[r];
        }
      } else if (!vx_view_copy_unpack(v, offset, vx_memref_sizes(d),
                                      (int64_t *)vx_memref_strides(d, ranks[k]),
                                      ranks[k])) {
        fprintf(stderr,
                "[Vx] FATAL: a region published a view of a tensor that Vx "
                "copied for the call, and Vx cannot give that view back in "
                "the tensor's own layout\n");
        abort();
      }
    }
  }
}

#endif /* VX_VIEW_COPY_H */
