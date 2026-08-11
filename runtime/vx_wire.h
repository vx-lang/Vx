//===- vx_wire.h - Encoding a dispatch for another machine ------*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The four messages a remote worker understands, and the bytes they are.
//
// See docs/discussions/implementation_plans/remote_dispatch_marshalling.md.
// The design decisions that show up here as code:
//
//   * **The ABI header is the schema.** An argument's shape on the wire is
//     decided by `arg_tags[i]`, which the compiler already emits and which
//     vx_dispatch_plan.h already decodes. A schema compiler would be a second
//     place the argument layout is written down, and the two would drift.
//   * **Buffers do not travel with a dispatch.** They arrive once by TRANSFER
//     and are named afterwards by a handle (vx_remote_region.h). Shipping
//     operands per dispatch would send llama2's `wq` 384 times in a 64-token
//     generation.
//   * **Little-endian, not negotiated.** Every target in the fleet is
//     little-endian, and a negotiation that has never been exercised is a bug
//     waiting rather than portability.
//
// Every read is bounds-checked against the end of the buffer and returns 0
// rather than reading past it. A truncated or malformed message is a thing that
// will happen -- a short read on a socket is the normal case, not the
// exceptional one -- and the difference between refusing it and walking off the
// end is the difference between a diagnostic and a crash in the wrong place.
//
// Vendor-free and header-only, so it is tested on any machine with no GPU and
// no network (tests/runtime/wire_test.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef VX_WIRE_H
#define VX_WIRE_H

#include "../include/vx_hardware_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/// 'V' 'X' 'R' 'M' -- present so a stray buffer is rejected at the first field
/// rather than interpreted as a length.
#define VX_WIRE_MAGIC UINT32_C(0x4D525856)
#define VX_WIRE_VERSION 1

enum {
  VX_WIRE_TRANSFER = 1, /* host -> worker: bytes; worker -> host: a handle */
  VX_WIRE_DISPATCH = 2, /* host -> worker: payload + args; back: slot results */
  VX_WIRE_FREE = 3,     /* host -> worker: a handle to release */
  VX_WIRE_FETCH = 4     /* host -> worker: a handle; back: its bytes */
};

/* --- cursors ------------------------------------------------------------- */

typedef struct {
  uint8_t *data;
  size_t capacity;
  size_t len;
  int overflow; /* set once a write did not fit; never partially written */
} vx_wire_writer;

typedef struct {
  const uint8_t *data;
  size_t len;
  size_t pos;
  int underflow; /* set once a read ran past the end */
} vx_wire_reader;

static inline void vx_wire_writer_init(vx_wire_writer *w, void *buf,
                                       size_t capacity) {
  w->data = (uint8_t *)buf;
  w->capacity = capacity;
  w->len = 0;
  w->overflow = 0;
}

static inline void vx_wire_reader_init(vx_wire_reader *r, const void *buf,
                                       size_t len) {
  r->data = (const uint8_t *)buf;
  r->len = len;
  r->pos = 0;
  r->underflow = 0;
}

static inline void vx_wire_put(vx_wire_writer *w, const void *src, size_t n) {
  if (w->overflow || w->len + n > w->capacity) {
    w->overflow = 1;
    return;
  }
  memcpy(w->data + w->len, src, n);
  w->len += n;
}

/// Reads are little-endian byte by byte rather than a cast, so the encoding
/// does not depend on the reader's alignment rules or its own endianness.
static inline int vx_wire_get(vx_wire_reader *r, void *dst, size_t n) {
  if (r->underflow || r->pos + n > r->len) {
    r->underflow = 1;
    return 0;
  }
  memcpy(dst, r->data + r->pos, n);
  r->pos += n;
  return 1;
}

static inline void vx_wire_put_u32(vx_wire_writer *w, uint32_t v) {
  uint8_t b[4];
  b[0] = (uint8_t)(v);
  b[1] = (uint8_t)(v >> 8);
  b[2] = (uint8_t)(v >> 16);
  b[3] = (uint8_t)(v >> 24);
  vx_wire_put(w, b, 4);
}

static inline int vx_wire_get_u32(vx_wire_reader *r, uint32_t *out) {
  uint8_t b[4];
  if (!vx_wire_get(r, b, 4)) {
    return 0;
  }
  *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
  return 1;
}

static inline void vx_wire_put_u64(vx_wire_writer *w, uint64_t v) {
  vx_wire_put_u32(w, (uint32_t)(v & 0xFFFFFFFFu));
  vx_wire_put_u32(w, (uint32_t)(v >> 32));
}

static inline int vx_wire_get_u64(vx_wire_reader *r, uint64_t *out) {
  uint32_t lo, hi;
  if (!vx_wire_get_u32(r, &lo) || !vx_wire_get_u32(r, &hi)) {
    return 0;
  }
  *out = (uint64_t)lo | ((uint64_t)hi << 32);
  return 1;
}

static inline void vx_wire_put_i32(vx_wire_writer *w, int32_t v) {
  vx_wire_put_u32(w, (uint32_t)v);
}

static inline int vx_wire_get_i32(vx_wire_reader *r, int32_t *out) {
  uint32_t v;
  if (!vx_wire_get_u32(r, &v)) {
    return 0;
  }
  *out = (int32_t)v;
  return 1;
}

static inline void vx_wire_put_i64(vx_wire_writer *w, int64_t v) {
  vx_wire_put_u64(w, (uint64_t)v);
}

static inline int vx_wire_get_i64(vx_wire_reader *r, int64_t *out) {
  uint64_t v;
  if (!vx_wire_get_u64(r, &v)) {
    return 0;
  }
  *out = (int64_t)v;
  return 1;
}

/* --- framing ------------------------------------------------------------- */

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t type;
  uint64_t body_len;
} vx_wire_header;

static inline void vx_wire_put_header(vx_wire_writer *w, uint32_t type,
                                      uint64_t body_len) {
  vx_wire_put_u32(w, VX_WIRE_MAGIC);
  vx_wire_put_u32(w, VX_WIRE_VERSION);
  vx_wire_put_u32(w, type);
  vx_wire_put_u64(w, body_len);
}

/// Refuses on a bad magic, an unknown version, or a body length that the buffer
/// cannot contain -- the last being the field an attacker or a bug would use to
/// make a reader allocate or walk past the end.
static inline int vx_wire_get_header(vx_wire_reader *r, vx_wire_header *out) {
  if (!vx_wire_get_u32(r, &out->magic) || !vx_wire_get_u32(r, &out->version) ||
      !vx_wire_get_u32(r, &out->type) || !vx_wire_get_u64(r, &out->body_len)) {
    return 0;
  }
  if (out->magic != VX_WIRE_MAGIC || out->version != VX_WIRE_VERSION) {
    return 0;
  }
  if (out->body_len > r->len - r->pos) {
    return 0;
  }
  return 1;
}

/* --- arguments ----------------------------------------------------------- */

/// One argument as it travels. Which union-ish half is meaningful is decided by
/// `tag`, exactly as it is in a local dispatch.
typedef struct {
  int32_t tag;

  /* scalars: the value, in as many bytes as its kind takes */
  uint8_t scalar[8];

  /* memrefs and slots */
  uint64_t handle;
  int32_t rank;
  int64_t sizes[VX_ABI_MAX_RANK];
  int64_t strides[VX_ABI_MAX_RANK];
} vx_wire_arg;

/// Encode one argument.
///
/// A memref sends its handle and its extents but *not* its bytes: the buffer is
/// already resident, put there by an earlier TRANSFER. Extents travel because a
/// view over a resident buffer has its own, and a strided sub-view (#344) will
/// have its own strides.
///
/// A slot sends a handle of 0 when it is storage the worker will publish into,
/// and a real handle when it already *holds* a buffer.
///
/// Both happen. `c = a @ b` gives a slot the kernel allocates through, and that
/// is the case the slot bit was introduced for. But a local tensor also lives
/// in a slot, so `matmul_into(&mut y, ..)` on one reaches its buffer through
/// that indirection -- `outkind=buffer` with a slot-tagged argument, which
/// vx_dispatch_plan.h already handles by following it. Sending nothing for a
/// slot meant the worker rebuilt an empty one and decoded a result of shape
/// 0x0, so the dispatch was refused with nothing to indicate why.
static inline void vx_wire_put_arg(vx_wire_writer *w, const vx_wire_arg *a) {
  int32_t kind = VX_ABI_KIND(a->tag);
  vx_wire_put_i32(w, a->tag);

  if (VX_ABI_IS_SLOT(a->tag)) {
    vx_wire_put_u64(w, a->handle);
    if (a->handle != 0) {
      vx_wire_put_i32(w, a->rank);
      for (int32_t i = 0; i < a->rank; ++i) {
        vx_wire_put_i64(w, a->sizes[i]);
      }
      for (int32_t i = 0; i < a->rank; ++i) {
        vx_wire_put_i64(w, a->strides[i]);
      }
    }
    return;
  }
  if (kind != VX_ABI_KIND_MEMREF) {
    size_t n = vx_dtype_bytes(kind);
    vx_wire_put(w, a->scalar, n ? n : 8);
    return;
  }

  vx_wire_put_u64(w, a->handle);
  vx_wire_put_i32(w, a->rank);
  for (int32_t i = 0; i < a->rank; ++i) {
    vx_wire_put_i64(w, a->sizes[i]);
  }
  for (int32_t i = 0; i < a->rank; ++i) {
    vx_wire_put_i64(w, a->strides[i]);
  }
}

/// Decode one argument, refusing anything it cannot represent.
///
/// The rank check is the one that matters: rank arrives from the wire and
/// indexes fixed arrays, so an unchecked value writes past them. It is bounded
/// by the same VX_ABI_MAX_RANK the tag encoding is, and a mismatch between the
/// tag's rank and the body's is refused rather than reconciled.
static inline int vx_wire_get_arg(vx_wire_reader *r, vx_wire_arg *out) {
  int32_t kind;
  memset(out, 0, sizeof(*out));

  if (!vx_wire_get_i32(r, &out->tag)) {
    return 0;
  }
  kind = VX_ABI_KIND(out->tag);

  if (VX_ABI_IS_SLOT(out->tag)) {
    if (!vx_wire_get_u64(r, &out->handle)) {
      return 0;
    }
    if (out->handle == 0) {
      return 1; /* storage to publish into */
    }
    if (!vx_wire_get_i32(r, &out->rank) || out->rank < 0 ||
        out->rank > VX_ABI_MAX_RANK) {
      return 0;
    }
    for (int32_t i = 0; i < out->rank; ++i) {
      if (!vx_wire_get_i64(r, &out->sizes[i])) {
        return 0;
      }
    }
    for (int32_t i = 0; i < out->rank; ++i) {
      if (!vx_wire_get_i64(r, &out->strides[i])) {
        return 0;
      }
    }
    return 1;
  }
  if (kind != VX_ABI_KIND_MEMREF) {
    size_t n = vx_dtype_bytes(kind);
    if (n == 0) {
      return 0; /* a scalar kind with no width is not a scalar we can carry */
    }
    return vx_wire_get(r, out->scalar, n);
  }

  if (!vx_wire_get_u64(r, &out->handle) || !vx_wire_get_i32(r, &out->rank)) {
    return 0;
  }
  if (out->rank < 0 || out->rank > VX_ABI_MAX_RANK ||
      out->rank != (int32_t)VX_ABI_RANK(out->tag)) {
    return 0;
  }
  for (int32_t i = 0; i < out->rank; ++i) {
    if (!vx_wire_get_i64(r, &out->sizes[i])) {
      return 0;
    }
  }
  for (int32_t i = 0; i < out->rank; ++i) {
    if (!vx_wire_get_i64(r, &out->strides[i])) {
      return 0;
    }
  }
  return 1;
}

/* --- messages ------------------------------------------------------------ */

/// TRANSFER: the bytes of a buffer, and what shape they are.
///
/// The reply carries the handle, because the worker mints it -- see
/// vx_remote_region.h on why identity is (worker, address) and cannot be agreed
/// on without coordinating.
static inline void vx_wire_put_transfer(vx_wire_writer *w, int32_t dtype,
                                        int32_t rank, const int64_t *sizes,
                                        const void *bytes, uint64_t nbytes) {
  vx_wire_put_i32(w, dtype);
  vx_wire_put_i32(w, rank);
  for (int32_t i = 0; i < rank; ++i) {
    vx_wire_put_i64(w, sizes[i]);
  }
  vx_wire_put_u64(w, nbytes);
  vx_wire_put(w, bytes, (size_t)nbytes);
}

typedef struct {
  int32_t dtype;
  int32_t rank;
  int64_t sizes[VX_ABI_MAX_RANK];
  uint64_t nbytes;
  const uint8_t *bytes; /* points into the reader's buffer; not copied */
} vx_wire_transfer;

static inline int vx_wire_get_transfer(vx_wire_reader *r,
                                       vx_wire_transfer *out) {
  memset(out, 0, sizeof(*out));
  if (!vx_wire_get_i32(r, &out->dtype) || !vx_wire_get_i32(r, &out->rank)) {
    return 0;
  }
  if (out->rank < 0 || out->rank > VX_ABI_MAX_RANK) {
    return 0;
  }
  for (int32_t i = 0; i < out->rank; ++i) {
    if (!vx_wire_get_i64(r, &out->sizes[i])) {
      return 0;
    }
  }
  if (!vx_wire_get_u64(r, &out->nbytes)) {
    return 0;
  }
  /* The length field decides how far the payload pointer reaches, so it is
     checked against what is actually present before it is believed. */
  if (out->nbytes > r->len - r->pos) {
    return 0;
  }
  out->bytes = r->data + r->pos;
  r->pos += (size_t)out->nbytes;
  return 1;
}

/// DISPATCH: the payload blob the compiler emitted, verbatim, then the
/// arguments. The blob is passed through untouched -- `kind=`, `roles=`,
/// `outkind=`, `topo=` and `toponame=` are decoded on the worker by the same
/// vx_dispatch_plan.h a local plugin uses, so there is exactly one decoder.
static inline void vx_wire_put_dispatch(vx_wire_writer *w, const void *payload,
                                        uint64_t payload_len,
                                        const vx_wire_arg *args,
                                        int64_t num_args) {
  vx_wire_put_u64(w, payload_len);
  vx_wire_put(w, payload, (size_t)payload_len);
  vx_wire_put_i64(w, num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    vx_wire_put_arg(w, &args[i]);
  }
}

typedef struct {
  const uint8_t *payload;
  uint64_t payload_len;
  int64_t num_args;
} vx_wire_dispatch;

/// Reads the fixed part; the caller then pulls `num_args` arguments with
/// vx_wire_get_arg, so a message with more arguments than the caller can hold
/// is refused by the caller rather than by a limit invented here.
static inline int vx_wire_get_dispatch(vx_wire_reader *r,
                                       vx_wire_dispatch *out) {
  memset(out, 0, sizeof(*out));
  if (!vx_wire_get_u64(r, &out->payload_len)) {
    return 0;
  }
  if (out->payload_len > r->len - r->pos) {
    return 0;
  }
  out->payload = r->data + r->pos;
  r->pos += (size_t)out->payload_len;

  if (!vx_wire_get_i64(r, &out->num_args) || out->num_args < 0) {
    return 0;
  }
  /* An argument is at least its tag, so a count that could not fit even that
     many tags is a lie and is refused before anything is allocated for it. */
  if ((uint64_t)out->num_args * 4 > r->len - r->pos) {
    return 0;
  }
  return 1;
}

/// What a dispatch produced: one entry per slot argument, in the order the
/// slots appeared.
///
/// A slot is the storage a kernel publishes an allocated result through, and
/// that storage is on the *host* -- it is the caller's own memory, which the
/// outlined kernel would have stored a descriptor into. So the result cannot
/// come back by the worker writing through it; the worker has nothing that
/// points there.
///
/// Instead the reply names the result and the host writes the descriptor
/// itself, with the handle where a pointer would be. That descriptor is then
/// exactly as usable as a local one -- it can be passed to the next dispatch,
/// which sends the handle onward -- and exactly as unusable by host code, which
/// faults on the non-canonical address rather than reading a wrong answer out
/// of it. See vx_remote_region.h.
typedef struct {
  uint64_t handle;
  int32_t rank;
  int64_t sizes[VX_ABI_MAX_RANK];
} vx_wire_result;

static inline void vx_wire_put_results(vx_wire_writer *w, int32_t status,
                                       const vx_wire_result *results,
                                       int64_t count) {
  vx_wire_put_i32(w, status);
  vx_wire_put_i64(w, count);
  for (int64_t i = 0; i < count; ++i) {
    vx_wire_put_u64(w, results[i].handle);
    vx_wire_put_i32(w, results[i].rank);
    for (int32_t j = 0; j < results[i].rank; ++j) {
      vx_wire_put_i64(w, results[i].sizes[j]);
    }
  }
}

static inline int vx_wire_get_results_header(vx_wire_reader *r, int32_t *status,
                                             int64_t *count) {
  if (!vx_wire_get_i32(r, status) || !vx_wire_get_i64(r, count)) {
    return 0;
  }
  if (*count < 0 || (uint64_t)*count * 12 > r->len - r->pos) {
    return 0;
  }
  return 1;
}

static inline int vx_wire_get_result(vx_wire_reader *r, vx_wire_result *out) {
  memset(out, 0, sizeof(*out));
  if (!vx_wire_get_u64(r, &out->handle) || !vx_wire_get_i32(r, &out->rank)) {
    return 0;
  }
  if (out->rank < 0 || out->rank > VX_ABI_MAX_RANK) {
    return 0;
  }
  for (int32_t i = 0; i < out->rank; ++i) {
    if (!vx_wire_get_i64(r, &out->sizes[i])) {
      return 0;
    }
  }
  return 1;
}

/// FETCH: read a resident buffer back.
///
/// The remote counterpart of `vx_plugin_transfer_device_to_host`, and it was
/// missing from the first three message types -- an omission that survived the
/// design document and the in-process round trip, because in one process the
/// "host" could simply read the worker's memory. Across a socket it cannot, and
/// a serving program has to: llama2 samples from the logits, so the last
/// dispatch of every token produces a value the host must actually see.
///
/// The reply is the bytes and nothing else; the frame already carries how many.
static inline void vx_wire_put_fetch(vx_wire_writer *w, uint64_t handle,
                                     uint64_t nbytes) {
  vx_wire_put_u64(w, handle);
  vx_wire_put_u64(w, nbytes);
}

static inline int vx_wire_get_fetch(vx_wire_reader *r, uint64_t *handle,
                                    uint64_t *nbytes) {
  return vx_wire_get_u64(r, handle) && vx_wire_get_u64(r, nbytes);
}

static inline void vx_wire_put_free(vx_wire_writer *w, uint64_t handle) {
  vx_wire_put_u64(w, handle);
}

static inline int vx_wire_get_free(vx_wire_reader *r, uint64_t *handle) {
  return vx_wire_get_u64(r, handle);
}

#endif /* VX_WIRE_H */
