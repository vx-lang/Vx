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

#include "vx_manifest.h"
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

static inline int vx_remote_free(int fd, uint64_t handle) {
  uint8_t request[16];
  vx_wire_writer w;
  vx_wire_writer_init(&w, request, sizeof(request));
  vx_wire_put_free(&w, handle);
  return vx_transport_send(fd, VX_WIRE_FREE, request, w.len) == VX_TRANSPORT_OK;
}

#endif /* VX_REMOTE_CLIENT_H */
