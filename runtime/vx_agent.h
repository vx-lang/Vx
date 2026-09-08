//===- vx_agent.h - The worker's half of a remote dispatch ------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Turning a dispatch that arrived over the wire back into the arguments a
// plugin already knows how to run.
//
// The point of this file is what it does *not* add. A worker that decoded
// `kind=`, `roles=` and `outkind=` itself would be a second implementation of
// runtime/vx_dispatch_plan.h, and the two would drift -- which is the same
// argument vx_host_call.h records about the ABI-tag mapping that had been
// copied into three backends. So instead the arriving arguments are rebuilt
// into exactly the `device_args` / `arg_tags` pair a local dispatch passes, and
// the existing decoder runs on them unchanged. A remote worker and a local
// plugin then disagree about nothing, because there is only one decoder.
//
// What rebuilding means concretely: a memref arrived as a handle plus extents,
// so it becomes a descriptor pointing at the resolved backing memory. A scalar
// arrived as its bytes. A slot arrived as a tag alone, so it becomes descriptor
// storage the worker will publish through, and whose contents travel back in
// the reply rather than being written through a pointer the worker does not
// have.
//
// Vendor-free and header-only, so the whole path is exercised on a machine with
// no GPU and no network (tests/runtime/loopback_test.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef VX_AGENT_H
#define VX_AGENT_H

#include "vx_remote_region.h"
#include "vx_wire.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/// Bytes an MLIR memref descriptor of this rank occupies:
/// { allocated, aligned, offset, sizes[rank], strides[rank] }.
static inline size_t vx_agent_desc_bytes(int32_t rank) {
  return 2 * sizeof(void *) + sizeof(int64_t) +
         2 * (size_t)rank * sizeof(int64_t);
}

/// Write a descriptor with the extents that arrived, rather than derived ones.
///
/// `vx_memref_write_desc` derives row-major strides, which is right for a
/// buffer the plugin just allocated and wrong for a view: a strided sub-view of
/// a resident tensor (#344) has strides that are a fact about the caller's
/// tensor, not about this allocation. So the wire's strides are honoured.
static inline void vx_agent_write_desc(void *desc, void *data, int32_t rank,
                                       const int64_t *sizes,
                                       const int64_t *strides) {
  void **ptrs = (void **)desc;
  int64_t *fields = (int64_t *)((char *)desc + 2 * sizeof(void *));
  ptrs[0] = data;
  ptrs[1] = data;
  fields[0] = 0; /* offset: the resolved pointer already includes it */
  for (int32_t i = 0; i < rank; ++i) {
    fields[1 + i] = sizes[i];
    fields[1 + rank + i] = strides[i];
  }
}

/// Storage a rebuilt argument list needs, supplied by the caller so this header
/// allocates nothing.
///
/// `desc_ptrs` exists because of an indirection that is easy to miss and silent
/// when missed: `device_args[i]` does not point at a descriptor, it points at a
/// *pointer* to one. `vx_operand_desc` dereferences twice
/// (`*(const void **)device_args[idx]`), which is the MLIR C interface's packed
/// form. Handing it a descriptor directly makes it read the first eight bytes
/// of that descriptor as an address.
typedef struct {
  void **device_args; /* [num_args], each pointing into desc_ptrs or scalars */
  void **desc_ptrs;   /* [num_args], the pointer device_args[i] points at */
  int32_t *arg_tags;  /* [num_args] */
  uint8_t *descriptors; /* descriptor bytes, laid out by the rebuild */
  size_t descriptors_capacity;
} vx_agent_args;

/// Rebuild one argument. Returns 0 if it cannot be honoured -- an unresolvable
/// handle, a rank that does not fit, or storage that has run out.
///
/// `desc_cursor` walks `storage->descriptors`; the caller starts it at zero.
static inline int vx_agent_rebuild_arg(const vx_remote_table *table,
                                       const vx_wire_arg *arg, int64_t index,
                                       vx_agent_args *storage,
                                       size_t *desc_cursor) {
  int32_t kind = VX_ABI_KIND(arg->tag);
  storage->arg_tags[index] = arg->tag;

  if (VX_ABI_IS_SLOT(arg->tag)) {
    /* A slot is `memref<memref<...>>`: an outer descriptor whose `aligned`
       points at the storage a result descriptor gets published into. Both
       halves have to exist here, because the worker is standing in for the
       kernel that would have allocated them.

       The inner storage starts zeroed, so a slot that was never published reads
       as a null data pointer rather than as stale bytes that happen to look
       like an answer. */
    int32_t held = (int32_t)VX_ABI_RANK(arg->tag);
    size_t inner = vx_agent_desc_bytes(held);
    size_t outer = vx_agent_desc_bytes(0);
    uint8_t *inner_p, *outer_p;

    if (held > VX_ABI_MAX_RANK ||
        *desc_cursor + inner + outer > storage->descriptors_capacity) {
      return 0;
    }
    inner_p = storage->descriptors + *desc_cursor;
    outer_p = inner_p + inner;
    memset(inner_p, 0, inner + outer);
    ((void **)outer_p)[0] = inner_p;
    ((void **)outer_p)[1] = inner_p;

    /* A slot that already holds a buffer -- a local tensor's, reached through
       the indirection it lives in -- arrives with a handle. The inner
       descriptor then names that buffer rather than waiting to be published
       into. */
    if (arg->handle != 0) {
      vx_remote_ref held;
      if (!vx_remote_resolve(table, arg->handle, &held)) {
        return 0;
      }
      vx_agent_write_desc(inner_p, (char *)held.region->remote + held.offset,
                          arg->rank, arg->sizes, arg->strides);
    }

    storage->desc_ptrs[index] = outer_p;
    storage->device_args[index] = &storage->desc_ptrs[index];
    *desc_cursor += inner + outer;
    return 1;
  }

  if (kind != VX_ABI_KIND_MEMREF) {
    /* A scalar is passed by address, the way libffi receives it locally: one
       indirection, not two. The bytes live in the wire argument, which outlives
       the call. */
    storage->device_args[index] = (void *)(uintptr_t)arg->scalar;
    return 1;
  }

  {
    vx_remote_ref ref;
    size_t need = vx_agent_desc_bytes(arg->rank);
    char *data;

    /* The bounds check in vx_remote_resolve is what stops an offset that ran
       past its region from being served out of the next one. Refusing here,
       naming nothing, is deliberate: the caller reports the failure with the
       worker and the offending handle, which a silent substitution would not
       allow. */
    if (!vx_remote_resolve(table, arg->handle, &ref)) {
      return 0;
    }
    if (arg->rank > VX_ABI_MAX_RANK ||
        *desc_cursor + need > storage->descriptors_capacity) {
      return 0;
    }
    data = (char *)ref.region->remote + ref.offset;
    vx_agent_write_desc(storage->descriptors + *desc_cursor, data, arg->rank,
                        arg->sizes, arg->strides);
    storage->desc_ptrs[index] = storage->descriptors + *desc_cursor;
    storage->device_args[index] = &storage->desc_ptrs[index];
    *desc_cursor += need;
    return 1;
  }
}

/// Rebuild a whole argument list. On success `storage->device_args` and
/// `storage->arg_tags` are exactly what `vx_plugin_dispatch_async` would have
/// been handed locally, and `vx_gemm_plan_decode` accepts them unchanged.
static inline int vx_agent_rebuild_args(const vx_remote_table *table,
                                        const vx_wire_arg *args,
                                        int64_t num_args,
                                        vx_agent_args *storage) {
  size_t cursor = 0;
  for (int64_t i = 0; i < num_args; ++i) {
    if (!vx_agent_rebuild_arg(table, &args[i], i, storage, &cursor)) {
      return 0;
    }
  }
  return 1;
}

/// Collect what the dispatch published, for the reply.
///
/// A slot's descriptor was written by whatever ran the kernel -- locally that
/// is `vx_gemm_publish_slot`, and the worker uses the same call. What the host
/// needs back is the handle for the buffer and its shape, not the descriptor,
/// since the pointer inside it means nothing on the other machine.
static inline int64_t
vx_agent_collect_results(const vx_remote_table *table, const vx_wire_arg *args,
                         int64_t num_args, vx_agent_args *storage,
                         vx_wire_result *out, int64_t out_capacity) {
  int64_t count = 0;
  for (int64_t i = 0; i < num_args; ++i) {
    if (!VX_ABI_IS_SLOT(args[i].tag)) {
      continue;
    }
    if (count >= out_capacity) {
      return -1;
    }
    {
      /* The same two hops the decoder took: the outer descriptor's `aligned`
         is the inner storage, and that is where the result descriptor was
         published. */
      const void *outer = storage->desc_ptrs[i];
      const void *inner = vx_memref_aligned(outer);
      int32_t rank = (int32_t)VX_ABI_RANK(args[i].tag);
      const int64_t *sizes = vx_memref_sizes(inner);
      void *published = vx_memref_aligned(inner);
      vx_remote_ref ref;

      /* The worker allocated the result in its own memory and published a
         pointer to it. Turning that back into a handle is a reverse lookup, and
         a published pointer that names no region is a bug in whatever ran the
         kernel rather than something to paper over. */
      out[count].handle = 0;
      for (size_t r = 0; r < table->count; ++r) {
        const vx_remote_region *reg = &table->regions[r];
        char *base = (char *)reg->remote;
        if (reg->live && published >= (void *)base &&
            published < (void *)(base + reg->size)) {
          out[count].handle = reg->base + (uint64_t)((char *)published - base);
          break;
        }
      }
      if (out[count].handle == 0 ||
          !vx_remote_resolve(table, out[count].handle, &ref)) {
        return -1;
      }

      out[count].rank = rank;
      for (int32_t j = 0; j < rank; ++j) {
        out[count].sizes[j] = sizes[j];
      }
      ++count;
    }
  }
  return count;
}

/// Apply a reply to the host's own slots.
///
/// The descriptor written here holds a *handle* where a pointer would be. That
/// is the whole trick: the next dispatch that passes this slot forward sends
/// the handle onward and the buffer never moves, while any host code that reads
/// through it faults on a non-canonical address instead of returning a wrong
/// answer.
static inline void vx_agent_apply_result(void *host_slot_desc,
                                         const vx_wire_result *result) {
  int64_t strides[VX_ABI_MAX_RANK];
  int64_t acc = 1;
  for (int32_t i = result->rank - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= result->sizes[i];
  }
  vx_agent_write_desc(host_slot_desc, (void *)(uintptr_t)result->handle,
                      result->rank, result->sizes, strides);
}

#endif /* VX_AGENT_H */
