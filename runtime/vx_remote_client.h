//===- vx_remote_client.h - The host's half of a remote dispatch *- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Connecting to a worker named in the manifest, and the four operations a
// plugin performs against it.
//
// The shape mirrors the local plugin ABI deliberately, one call each:
//
//   vx_plugin_alloc_and_transfer   ->  TRANSFER
//   vx_plugin_dispatch_async       ->  DISPATCH
//   vx_plugin_transfer_device_to_host -> FETCH
//   vx_plugin_free                 ->  FREE
//
// so a backend routes by choosing which of two implementations to call rather
// than by restructuring anything. That is the whole of location transparency at
// the runtime layer: the same operation, a different provider.
//
// Connections are made once per worker and kept. A dispatch is already a round
// trip; adding a TCP handshake to each one would make the per-token cost of a
// disaggregated Llama the connection rather than the arithmetic.
//
// **What this does not do is decide.** Whether a topology is remote at all is
// the manifest's answer (vx_manifest.h), and a name that is absent means local.
// Nothing here should ever be reached for a topology that lives on this
// machine.
//
//===----------------------------------------------------------------------===//

#ifndef VX_REMOTE_CLIENT_H
#define VX_REMOTE_CLIENT_H

#include "vx_agent.h"
#include "vx_dispatch_plan.h"
#include "vx_manifest.h"
#include "vx_remote_region.h"
#include "vx_transport.h"
#include "vx_wire.h"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct {
  char name[VX_MANIFEST_MAX_NAME];
  int fd; /* -1 when not connected */
} vx_remote_conn;

#define VX_REMOTE_MAX_CONNS VX_MANIFEST_MAX_WORKERS

typedef struct {
  vx_remote_conn conns[VX_REMOTE_MAX_CONNS];
  size_t count;
} vx_remote_pool;

static inline void vx_remote_pool_init(vx_remote_pool *p) { p->count = 0; }

/// Open a connection, or return the one already open for this worker.
///
/// Returns -1 on failure, having said which worker and why on stderr. A
/// placement that names a machine which cannot be reached is not something to
/// degrade quietly into running locally: the program asked for a device and did
/// not get it, and a silent fallback would produce correct numbers while the
/// distribution under test did not happen.
static inline int vx_remote_connect(vx_remote_pool *pool,
                                    const vx_manifest_entry *w) {
  struct addrinfo hints, *res = NULL, *it;
  char port_s[16];
  int fd = -1;
  size_t i;

  for (i = 0; i < pool->count; ++i) {
    if (strcmp(pool->conns[i].name, w->name) == 0) {
      return pool->conns[i].fd;
    }
  }
  if (pool->count == VX_REMOTE_MAX_CONNS) {
    fprintf(stderr, "[Vx remote] FATAL: too many workers\n");
    return -1;
  }

  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  snprintf(port_s, sizeof(port_s), "%d", w->port);

  if (getaddrinfo(w->host, port_s, &hints, &res) != 0 || !res) {
    fprintf(stderr, "[Vx remote] FATAL: cannot resolve %s (%s:%d)\n", w->name,
            w->host, w->port);
    return -1;
  }
  for (it = res; it; it = it->ai_next) {
    fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (fd < 0) {
      continue;
    }
    if (connect(fd, it->ai_addr, it->ai_addrlen) == 0) {
      break;
    }
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);

  if (fd < 0) {
    fprintf(stderr, "[Vx remote] FATAL: cannot reach %s at %s:%d\n", w->name,
            w->host, w->port);
    return -1;
  }

  /* A dispatch is a request followed by a wait for its reply, so Nagle has
     nothing to coalesce and only adds latency to every one of them. */
  {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }

  snprintf(pool->conns[pool->count].name, VX_MANIFEST_MAX_NAME, "%s", w->name);
  pool->conns[pool->count].fd = fd;
  ++pool->count;
  return fd;
}

static inline void vx_remote_pool_close(vx_remote_pool *pool) {
  size_t i;
  for (i = 0; i < pool->count; ++i) {
    if (pool->conns[i].fd >= 0) {
      close(pool->conns[i].fd);
      pool->conns[i].fd = -1;
    }
  }
  pool->count = 0;
}

/// Stage a buffer onto the worker and get back the handle naming it.
/// Returns 0 on failure.
static inline uint64_t vx_remote_transfer(int fd, const void *bytes,
                                          uint64_t nbytes, int32_t dtype,
                                          uint8_t *scratch,
                                          size_t scratch_len) {
  const int64_t sizes[1] = {(int64_t)nbytes};
  vx_wire_writer w;
  vx_wire_reader r;
  uint32_t type = 0;
  uint64_t reply_len = 0, handle = 0;
  uint8_t reply[64];

  vx_wire_writer_init(&w, scratch, scratch_len);
  vx_wire_put_transfer(&w, dtype, 1, sizes, bytes, nbytes);
  if (w.overflow) {
    fprintf(stderr,
            "[Vx remote] FATAL: %llu bytes exceeds the staging buffer\n",
            (unsigned long long)nbytes);
    return 0;
  }
  if (vx_transport_send(fd, VX_WIRE_TRANSFER, scratch, w.len) !=
      VX_TRANSPORT_OK) {
    return 0;
  }
  if (vx_transport_recv(fd, &type, reply, sizeof(reply), &reply_len) !=
      VX_TRANSPORT_OK) {
    return 0;
  }
  vx_wire_reader_init(&r, reply, (size_t)reply_len);
  return vx_wire_get_u64(&r, &handle) ? handle : 0;
}

/// Read a resident buffer back. Returns 1 on success.
static inline int vx_remote_fetch(int fd, uint64_t handle, void *dst,
                                  uint64_t nbytes) {
  uint8_t request[32];
  vx_wire_writer w;
  uint32_t type = 0;
  uint64_t got = 0;
  int rc;

  vx_wire_writer_init(&w, request, sizeof(request));
  vx_wire_put_fetch(&w, handle, nbytes);
  if (vx_transport_send(fd, VX_WIRE_FETCH, request, w.len) != VX_TRANSPORT_OK) {
    return 0;
  }
  rc = vx_transport_recv(fd, &type, dst, nbytes, &got);
  /* An empty reply is the worker saying it could not honour the handle. Short
     of that, a length that is not what was asked for means the two sides
     disagree about the buffer, which is worse than a refusal. */
  return rc == VX_TRANSPORT_OK && got == nbytes;
}

/// Turn a local argument list into wire arguments.
///
/// The inverse of vx_agent.h's rebuild, and the reason a memref's *handle* is
/// simply the pointer inside its descriptor: a buffer staged onto a remote
/// worker had its handle returned by `vx_plugin_alloc_and_transfer` and stored
/// wherever the program keeps pointers, so by the time a dispatch names it, the
/// descriptor already holds it. Nothing has to be looked up.
///
/* Defined below; a dispatch releases what it staged for the call. */
static inline int vx_remote_free(int fd, uint64_t handle);

/// A memref whose pointer is not a handle names memory on *this* machine, so it
/// is staged onto the worker for the duration of the call and released after.
/// That is precisely what the local path does -- runtime/cuda_dispatch.cpp
/// stages every non-resident operand per dispatch -- so this is the same data
/// movement with framing around it, not a new cost.
///
/// It is what makes an ordinary program dispatchable at all. llama2.vx stages
/// its *weights* once and keeps them resident, which is the whole reason a
/// disaggregated decode is not a fax machine; but its activations and its
/// output buffers are host memory that changes every token, and refusing those
/// would mean only a program with no inputs and no outputs could be sent.
///
/// The obvious improvement is to carry small operands inline in the DISPATCH
/// rather than as a TRANSFER and a FREE around it, which trades three round
/// trips for a longer message. Left undone deliberately: it is an optimisation
/// of something that works, and the wire does not have to change for it.
static inline int vx_remote_encode_args(int fd, void **device_args,
                                        const int32_t *arg_tags,
                                        int64_t num_args, vx_wire_arg *out,
                                        uint64_t *staged, int64_t *num_staged,
                                        uint8_t *scratch, size_t scratch_len) {
  *num_staged = 0;
  for (int64_t i = 0; i < num_args; ++i) {
    int32_t tag = arg_tags[i];
    int32_t kind = VX_ABI_KIND(tag);
    memset(&out[i], 0, sizeof(out[i]));
    out[i].tag = tag;

    if (VX_ABI_IS_SLOT(tag)) {
      /* Two kinds of slot, told apart by what it currently holds. Empty means
         storage the worker will publish into. Non-empty means a local tensor's
         buffer reached through the indirection it lives in, which has to travel
         like any other operand or the worker decodes a result of shape 0x0. */
      const void *outer = *(const void **)device_args[i];
      const void *inner = outer ? vx_memref_aligned(outer) : NULL;
      void *held = inner ? vx_memref_aligned(inner) : NULL;
      if (!held) {
        continue;
      }
      {
        int32_t rank = (int32_t)VX_ABI_RANK(tag);
        int32_t elem = (int32_t)VX_ABI_ELEM(tag);
        const int64_t *sizes = vx_memref_sizes(inner);
        const int64_t *strides = vx_memref_strides(inner, rank);
        uint64_t handle = (uint64_t)(uintptr_t)vx_memref_data(inner, elem);
        if (!vx_remote_addr_is_handle(handle)) {
          uint64_t span =
              rank > 0 ? (uint64_t)sizes[0] * (uint64_t)strides[0] : 1;
          handle = vx_remote_transfer(fd, (const void *)(uintptr_t)handle,
                                      span * (uint64_t)vx_dtype_bytes(elem),
                                      elem, scratch, scratch_len);
          if (handle == 0) {
            return 0;
          }
          staged[(*num_staged)++] = handle;
        }
        out[i].handle = handle;
        out[i].rank = rank;
        for (int32_t d = 0; d < rank; ++d) {
          out[i].sizes[d] = sizes[d];
          out[i].strides[d] = strides[d];
        }
      }
      continue;
    }
    if (kind != VX_ABI_KIND_MEMREF) {
      size_t n = vx_dtype_bytes(kind);
      if (n == 0 || n > sizeof(out[i].scalar)) {
        return 0;
      }
      memcpy(out[i].scalar, device_args[i], n);
      continue;
    }

    {
      const void *desc = *(const void **)device_args[i];
      int32_t rank = (int32_t)VX_ABI_RANK(tag);
      int32_t elem = (int32_t)VX_ABI_ELEM(tag);
      const int64_t *sizes;
      const int64_t *strides;
      uint64_t handle;

      if (!desc || rank < 0 || rank > VX_ABI_MAX_RANK) {
        return 0;
      }
      sizes = vx_memref_sizes(desc);
      strides = vx_memref_strides(desc, rank);
      handle = (uint64_t)(uintptr_t)vx_memref_data(desc, elem);

      if (!vx_remote_addr_is_handle(handle)) {
        /* Local memory. Stage the whole extent the descriptor spans, so a
           strided view arrives with the rows it refers to rather than only the
           elements it touches, and its strides still mean what they said. */
        uint64_t span =
            rank > 0 ? (uint64_t)sizes[0] * (uint64_t)strides[0] : 1;
        uint64_t bytes = span * (uint64_t)vx_dtype_bytes(elem);
        handle = vx_remote_transfer(fd, (const void *)(uintptr_t)handle, bytes,
                                    elem, scratch, scratch_len);
        if (handle == 0) {
          return 0;
        }
        staged[(*num_staged)++] = handle;
      }

      out[i].handle = handle;
      out[i].rank = rank;
      for (int32_t d = 0; d < rank; ++d) {
        out[i].sizes[d] = sizes[d];
        out[i].strides[d] = strides[d];
      }
    }
  }
  return 1;
}

/// Send a dispatch and apply whatever it published back into the caller's
/// slots. Returns 0 when the worker declined, which the caller must treat as
/// "run it here" rather than as a failure.
static inline int vx_remote_dispatch(int fd, const void *payload,
                                     uint64_t payload_len, void **device_args,
                                     const int32_t *arg_tags, int64_t num_args,
                                     vx_wire_arg *args, uint8_t *scratch,
                                     size_t scratch_len) {
  vx_wire_writer w;
  vx_wire_reader r;
  uint8_t reply[4096];
  uint32_t type = 0;
  uint64_t reply_len = 0;
  int32_t status = -1;
  int64_t count = 0;

  uint64_t staged[64];
  int64_t num_staged = 0;

  if (!vx_remote_encode_args(fd, device_args, arg_tags, num_args, args, staged,
                             &num_staged, scratch, scratch_len)) {
    return 0;
  }

  vx_wire_writer_init(&w, scratch, scratch_len);
  vx_wire_put_dispatch(&w, payload, payload_len, args, num_args);
  if (w.overflow) {
    return 0;
  }
  if (vx_transport_send(fd, VX_WIRE_DISPATCH, scratch, w.len) !=
      VX_TRANSPORT_OK) {
    return 0;
  }
  if (vx_transport_recv(fd, &type, reply, sizeof(reply), &reply_len) !=
      VX_TRANSPORT_OK) {
    return 0;
  }

  vx_wire_reader_init(&r, reply, (size_t)reply_len);
  if (!vx_wire_get_results_header(&r, &status, &count) || status != 0) {
    return 0;
  }

  /* Results arrive in slot order, so they are applied in the same order the
     slots appear. A descriptor written here holds a handle where a pointer
     would be -- see vx_agent_apply_result. */
  for (int64_t i = 0; i < num_args && count > 0; ++i) {
    vx_wire_result res;
    if (!VX_ABI_IS_SLOT(arg_tags[i])) {
      continue;
    }
    if (!vx_wire_get_result(&r, &res)) {
      return 0;
    }
    vx_agent_apply_result(*(void **)device_args[i], &res);
    --count;
  }

  /* An `outkind=buffer` result was written into a buffer that lives here, so
     the worker filled the copy it was staged and the original still holds what
     it held before. Reading it back is the counterpart of staging it. */
  {
    const char *outkind =
        vx_payload_field(payload, (size_t)payload_len, "outkind=");
    if (outkind && strcmp(outkind, "buffer") == 0) {
      const char *roles =
          vx_payload_field(payload, (size_t)payload_len, "roles=");
      int ai = -1, bi = -1, oi = -1;
      if (roles && vx_parse_roles(roles, &ai, &bi, &oi) && oi >= 0 &&
          oi < num_args && args[oi].handle != 0) {
        const void *desc = *(const void **)device_args[oi];
        if (VX_ABI_IS_SLOT(arg_tags[oi])) {
          desc = vx_memref_aligned(desc);
        }
        int32_t elem = (int32_t)VX_ABI_ELEM(arg_tags[oi]);
        int32_t rank = (int32_t)VX_ABI_RANK(arg_tags[oi]);
        const int64_t *sizes = vx_memref_sizes(desc);
        const int64_t *strides = vx_memref_strides(desc, rank);
        uint64_t span =
            rank > 0 ? (uint64_t)sizes[0] * (uint64_t)strides[0] : 1;
        uint64_t bytes = span * (uint64_t)vx_dtype_bytes(elem);
        if (!vx_remote_fetch(fd, args[oi].handle, vx_memref_data(desc, elem),
                             bytes)) {
          return 0;
        }
      }
    }
  }

  for (int64_t i = 0; i < num_staged; ++i) {
    vx_remote_free(fd, staged[i]);
  }
  return 1;
}

/// Release a resident buffer, and wait for the acknowledgement.
///
/// The wait is not politeness. Every message here is a request with a reply,
/// and a client that skipped one would leave it in the stream for the *next*
/// read to consume -- so the reply to a later FETCH would be the stale FREE
/// ack, and that FETCH would report failure for a reason having nothing to do
/// with the handle it asked about. Exactly that happened while this was being
/// written: a test asserting an unknown handle is refused passed by reading a
/// leftover acknowledgement.
static inline int vx_remote_free(int fd, uint64_t handle) {
  uint8_t request[16];
  uint8_t reply[16];
  vx_wire_writer w;
  uint32_t type = 0;
  uint64_t reply_len = 0;

  vx_wire_writer_init(&w, request, sizeof(request));
  vx_wire_put_free(&w, handle);
  if (vx_transport_send(fd, VX_WIRE_FREE, request, w.len) != VX_TRANSPORT_OK) {
    return 0;
  }
  return vx_transport_recv(fd, &type, reply, sizeof(reply), &reply_len) ==
         VX_TRANSPORT_OK;
}

#endif /* VX_REMOTE_CLIENT_H */
