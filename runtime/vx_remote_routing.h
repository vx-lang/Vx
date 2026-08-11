//===- vx_remote_routing.h - Sending an operation elsewhere -----*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The decision every backend makes before doing anything itself: is the
// topology this operation names a machine somewhere else? (#348)
//
// Four `try` functions, one per plugin entry point, each answering "I handled
// it" or "it is yours". A backend gains four lines and no structure -- which is
// the point, because the alternative is each backend growing its own copy of
// the same decision, and runtime/vx_host_call.h already records what happens
// then.
//
// **Everything is local unless a manifest says otherwise.** With no
// VX_FLEET_MANIFEST set, or a manifest that does not name a topology, every one
// of these returns "not mine" immediately and the backend behaves exactly as it
// did before this file existed. That is what lets one program text be a
// single-machine run and a distributed one.
//
//===----------------------------------------------------------------------===//

#ifndef VX_REMOTE_ROUTING_H
#define VX_REMOTE_ROUTING_H

#include "vx_remote_client.h"

#include <stdlib.h>

namespace {

/// The fleet, read once from $VX_FLEET_MANIFEST.
///
/// A manifest that exists and cannot be parsed aborts rather than falling back
/// to local. Someone wrote that file meaning to distribute this program, and
/// running it all on one machine instead would produce correct output and a
/// wrong experiment -- with nothing to notice afterwards.
inline const vx_manifest &vx_routing_manifest() {
  static vx_manifest m = [] {
    vx_manifest loaded;
    const char *path = getenv("VX_FLEET_MANIFEST");
    int rc = vx_manifest_load(&loaded, path);
    if (rc < 0) {
      fprintf(stderr, "[Vx remote] FATAL: %s is not a readable manifest\n",
              path ? path : "(null)");
      abort();
    }
    if (rc > 0 && getenv("VX_DISPATCH_VERBOSE")) {
      for (size_t i = 0; i < loaded.count; ++i) {
        fprintf(stderr, "[Vx remote] %s -> %s:%d (topology %d)\n",
                loaded.workers[i].name, loaded.workers[i].host,
                loaded.workers[i].port, loaded.workers[i].dispatch_id);
      }
    }
    return loaded;
  }();
  return m;
}

inline vx_remote_pool &vx_routing_pool() {
  static vx_remote_pool pool = [] {
    vx_remote_pool p;
    vx_remote_pool_init(&p);
    return p;
  }();
  return pool;
}

/// Scratch for the one message in flight. Sized for a staged weight blob rather
/// than a token: llama2 transfers each projection in a single TRANSFER.
inline uint8_t *vx_routing_scratch(size_t *len) {
  static const size_t kLen = 64u << 20;
  static uint8_t *buf = (uint8_t *)malloc(kLen);
  *len = buf ? kLen : 0;
  return buf;
}

/// The worker a topology id names, or NULL for "local".
inline const vx_manifest_entry *vx_routing_worker(int32_t topology_id) {
  const vx_manifest &m = vx_routing_manifest();
  if (m.count == 0) {
    return NULL;
  }
  return vx_manifest_find_by_id(&m, topology_id);
}

/// The worker a dispatch names, preferring the name it carries.
///
/// `toponame=` is the identity and `topo=` is a hash of it, so the name is
/// asked first and the id is the fallback for a producer that did not send one
/// -- the flat path builds a spawn from an instruction stream that has the id
/// and not the name.
inline const vx_manifest_entry *
vx_routing_worker_for_payload(const void *payload, size_t size) {
  const vx_manifest &m = vx_routing_manifest();
  if (m.count == 0) {
    return NULL;
  }
  const char *name = vx_payload_field(payload, size, "toponame=");
  if (name) {
    const vx_manifest_entry *w = vx_manifest_find(&m, name);
    if (w) {
      return w;
    }
  }
  return vx_manifest_find_by_id(&m, vx_payload_topology(payload, size));
}

/// Connect, or abort. A placement that names a machine which cannot be reached
/// is not something to degrade quietly into a local run.
inline int vx_routing_fd(const vx_manifest_entry *w) {
  int fd = vx_remote_connect(&vx_routing_pool(), w);
  if (fd < 0) {
    abort();
  }
  return fd;
}

/// Returns 1 and sets `*out` when the allocation belongs to a remote worker.
inline int vx_routing_try_alloc(size_t bytes, void *host_ptr,
                                uint32_t topology_id, void **out) {
  const vx_manifest_entry *w = vx_routing_worker((int32_t)topology_id);
  if (!w) {
    return 0;
  }
  size_t scratch_len = 0;
  uint8_t *scratch = vx_routing_scratch(&scratch_len);
  uint64_t handle =
      vx_remote_transfer(vx_routing_fd(w), host_ptr, (uint64_t)bytes,
                         VX_DTYPE_UNKNOWN, scratch, scratch_len);
  if (handle == 0) {
    fprintf(stderr, "[Vx remote] FATAL: %s refused %zu bytes\n", w->name,
            bytes);
    abort();
  }
  *out = (void *)(uintptr_t)handle;
  return 1;
}

/// Returns 1 when the dispatch was served remotely.
///
/// A worker that declines returns 0, and the backend then runs the kernel
/// itself -- an unroutable kernel is a performance outcome on one machine and
/// stays one across two.
inline int vx_routing_try_dispatch(const void *payload, size_t payload_size,
                                   void **device_args, const int32_t *arg_tags,
                                   int64_t num_args) {
  const vx_manifest_entry *w =
      vx_routing_worker_for_payload(payload, payload_size);
  if (!w) {
    return 0;
  }
  static vx_wire_arg args[64];
  if (num_args > 64) {
    return 0;
  }
  size_t scratch_len = 0;
  uint8_t *scratch = vx_routing_scratch(&scratch_len);
  return vx_remote_dispatch(vx_routing_fd(w), payload, (uint64_t)payload_size,
                            device_args, arg_tags, num_args, args, scratch,
                            scratch_len);
}

/// Returns 1 when the read-back was served remotely.
inline int vx_routing_try_fetch(void *device_ptr, void *host_ptr, size_t bytes,
                                uint32_t topology_id) {
  const vx_manifest_entry *w = vx_routing_worker((int32_t)topology_id);
  if (!w || !vx_remote_addr_is_handle((uint64_t)(uintptr_t)device_ptr)) {
    return 0;
  }
  if (!vx_remote_fetch(vx_routing_fd(w), (uint64_t)(uintptr_t)device_ptr,
                       host_ptr, (uint64_t)bytes)) {
    fprintf(stderr, "[Vx remote] FATAL: %s could not return %zu bytes\n",
            w->name, bytes);
    abort();
  }
  return 1;
}

/// Returns 1 when the release was served remotely.
inline int vx_routing_try_free(void *device_ptr, uint32_t topology_id) {
  const vx_manifest_entry *w = vx_routing_worker((int32_t)topology_id);
  if (!w || !vx_remote_addr_is_handle((uint64_t)(uintptr_t)device_ptr)) {
    return 0;
  }
  vx_remote_free(vx_routing_fd(w), (uint64_t)(uintptr_t)device_ptr);
  return 1;
}

} // namespace

#endif /* VX_REMOTE_ROUTING_H */
