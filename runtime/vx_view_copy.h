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
// This header plans the spans and needs no vendor library, so
// tests/runtime/view_copy_test.cpp runs it on any machine.
//
//===----------------------------------------------------------------------===//

#ifndef VX_VIEW_COPY_H
#define VX_VIEW_COPY_H

#include "../include/vx_hardware_runtime.h"

#include <stddef.h>
#include <stdint.h>

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

  /* vx_view_copy_point_at fills these, and vx_view_copy_restore reads them. */
  void *allocated;
  void *aligned;
  int64_t offset;
} vx_view_copy;

/// Bytes that one or more views reach, copied as one block.
typedef struct {
  uintptr_t start; /* the address of the span's first byte */
  uint64_t bytes;
  void *copy; /* the caller's copy of the span */
} vx_copy_span;

/// The root of the group that holds view `i`. The group links run from a view
/// to a view with a smaller index, so the root is the group's first view.
static inline int64_t vx_view_copy_root(const vx_view_copy *views, int64_t i) {
  while (views[i].span != i) {
    i = views[i].span;
  }
  return i;
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
  }
  return 1;
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
  }
}

/// Undo vx_view_copy_point_at in every descriptor in `descs` that still
/// points at a view's copy. That includes the views' own descriptors, and a
/// descriptor that the region published from a view. A published descriptor
/// can have its own offset, so the view's old offset is added to it.
static inline void vx_view_copy_restore(const vx_view_copy *views, int64_t n,
                                        const vx_copy_span *spans,
                                        void *const *descs, int64_t num_descs) {
  for (int64_t i = 0; i < n; ++i) {
    const vx_view_copy *v = &views[i];
    void *at;
    if (v->span < 0) {
      continue;
    }
    at = (char *)spans[v->span].copy + v->at;
    for (int64_t k = 0; k < num_descs; ++k) {
      void **d = (void **)descs[k];
      if (d && d[1] == at) {
        d[0] = v->allocated;
        d[1] = v->aligned;
        *(int64_t *)((char *)d + 2 * sizeof(void *)) += v->offset;
      }
    }
  }
}

#endif /* VX_VIEW_COPY_H */
