//===- vx_transport.h - Getting a message across a socket -------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framed send and receive over a stream file descriptor.
//
// Small, and every line of it is a case that a naive version gets wrong. A
// stream socket is not a message socket: `write` may accept fewer bytes than it
// was given and `read` may return fewer than were asked for, at any boundary
// and for no reason the caller can see. Code that treats one call as one
// message works on a laptop and on small messages, and then fails on the first
// buffer larger than a socket's buffer -- which for a KV cache handoff is every
// message that matters.
//
// So: loops, `EINTR` retried, end-of-stream distinguished from error, and a
// body length that is *drained* rather than left in the socket when it does not
// fit the caller's buffer. That last one is the difference between one refused
// message and a stream that is desynchronised from then on, every subsequent
// header read out of the middle of a body.
//
// The framing itself is runtime/vx_wire.h's: magic, version, type, length. This
// header adds no format, only the discipline of getting those bytes onto and
// off a descriptor.
//
// POSIX only -- it is used where a worker is a separate process. Vendor-free,
// and exercised over a socketpair between a real forked child in
// tests/runtime/transport_test.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef VX_TRANSPORT_H
#define VX_TRANSPORT_H

#include "vx_wire.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

/// magic + version + type + body_len, as vx_wire_put_header writes them.
#define VX_TRANSPORT_HEADER_BYTES 20

enum {
  VX_TRANSPORT_OK = 0,
  VX_TRANSPORT_EOF = 1,   /* the peer closed cleanly, between messages */
  VX_TRANSPORT_ERROR = 2, /* the descriptor failed */
  VX_TRANSPORT_MALFORMED =
      3, /* magic or version wrong; the stream is not ours */
  VX_TRANSPORT_TOO_LARGE = 4 /* body drained, stream still usable */
};

/// Write every byte, or say why not.
///
/// The return of `write` is how many bytes it took, which is not necessarily
/// how many it was offered. Ignoring that is the classic way to send a
/// truncated message and then read the next header out of the middle of it.
///
/// Honest about its coverage: a blocking unix socket queues the whole body or
/// blocks, so tests/runtime/transport_test.cpp does not force this loop to go
/// round twice and removing it changes nothing there. It stays because a short
/// write is POSIX-legal and does occur on TCP under memory pressure, which is
/// the transport this exists for -- but it is unproven code until a test on a
/// real network proves it.
///
/// Blocking descriptors only: `EAGAIN` is treated as an error rather than
/// retried, because a non-blocking socket needs a poll loop and a caller that
/// expects one, not a spin here.
static inline int vx_transport_write_all(int fd, const void *buf, size_t len) {
  const char *p = (const char *)buf;
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = write(fd, p + sent, len - sent);
    if (n < 0) {
      if (errno == EINTR) {
        continue; /* a signal, not a failure */
      }
      return VX_TRANSPORT_ERROR;
    }
    if (n == 0) {
      return VX_TRANSPORT_ERROR;
    }
    sent += (size_t)n;
  }
  return VX_TRANSPORT_OK;
}

/// Read exactly `len` bytes. Returns VX_TRANSPORT_EOF only when the peer closed
/// before *any* of them arrived; a close partway through a message is an error,
/// because the message is gone either way but the two mean different things to
/// whoever is debugging.
static inline int vx_transport_read_all(int fd, void *buf, size_t len) {
  char *p = (char *)buf;
  size_t got = 0;
  while (got < len) {
    ssize_t n = read(fd, p + got, len - got);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return VX_TRANSPORT_ERROR;
    }
    if (n == 0) {
      return got == 0 ? VX_TRANSPORT_EOF : VX_TRANSPORT_ERROR;
    }
    got += (size_t)n;
  }
  return VX_TRANSPORT_OK;
}

/// Read and discard `len` bytes, so an unusable message does not desynchronise
/// everything after it.
static inline int vx_transport_drain(int fd, uint64_t len) {
  char scratch[1024];
  while (len > 0) {
    size_t chunk = len > sizeof(scratch) ? sizeof(scratch) : (size_t)len;
    int rc = vx_transport_read_all(fd, scratch, chunk);
    if (rc != VX_TRANSPORT_OK) {
      return rc;
    }
    len -= chunk;
  }
  return VX_TRANSPORT_OK;
}

/// Send one framed message.
static inline int vx_transport_send(int fd, uint32_t type, const void *body,
                                    uint64_t body_len) {
  uint8_t header[VX_TRANSPORT_HEADER_BYTES];
  vx_wire_writer w;
  int rc;

  vx_wire_writer_init(&w, header, sizeof(header));
  vx_wire_put_header(&w, type, body_len);
  if (w.overflow || w.len != sizeof(header)) {
    return VX_TRANSPORT_ERROR;
  }

  rc = vx_transport_write_all(fd, header, sizeof(header));
  if (rc != VX_TRANSPORT_OK) {
    return rc;
  }
  if (body_len == 0) {
    return VX_TRANSPORT_OK;
  }
  return vx_transport_write_all(fd, body, (size_t)body_len);
}

/// Receive one framed message into `body`.
///
/// On VX_TRANSPORT_TOO_LARGE the body has been drained and the stream is still
/// positioned at the next header, so a caller may keep going. On
/// VX_TRANSPORT_MALFORMED it has not been, and cannot be: a header whose magic
/// is wrong is a header whose length field means nothing, so there is no safe
/// number of bytes to skip and the connection has to be dropped.
static inline int vx_transport_recv(int fd, uint32_t *type, void *body,
                                    uint64_t capacity, uint64_t *body_len) {
  uint8_t header[VX_TRANSPORT_HEADER_BYTES];
  vx_wire_reader r;
  uint32_t magic = 0, version = 0;
  uint64_t len = 0;
  int rc = vx_transport_read_all(fd, header, sizeof(header));

  if (rc != VX_TRANSPORT_OK) {
    return rc;
  }

  /* Parsed by hand rather than with vx_wire_get_header, which additionally
     checks the body against a buffer it can see. Here the body has not been
     read yet, and its length is what decides how much to read. */
  vx_wire_reader_init(&r, header, sizeof(header));
  if (!vx_wire_get_u32(&r, &magic) || !vx_wire_get_u32(&r, &version) ||
      !vx_wire_get_u32(&r, type) || !vx_wire_get_u64(&r, &len)) {
    return VX_TRANSPORT_ERROR;
  }
  if (magic != VX_WIRE_MAGIC || version != VX_WIRE_VERSION) {
    return VX_TRANSPORT_MALFORMED;
  }

  *body_len = len;
  if (len > capacity) {
    int drained = vx_transport_drain(fd, len);
    return drained == VX_TRANSPORT_OK ? VX_TRANSPORT_TOO_LARGE : drained;
  }
  if (len == 0) {
    return VX_TRANSPORT_OK;
  }
  return vx_transport_read_all(fd, body, (size_t)len);
}

#endif /* VX_TRANSPORT_H */
