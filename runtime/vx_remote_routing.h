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

#include <mutex>
#include <stdlib.h>
#include <string.h>

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

/// Say, once per placement, that a manifest was loaded and does not mention it.
///
/// Not an error: a manifest may list one of two GPUs and mean "the other is
/// local", which the GEMM benchmark's loopback manifest does deliberately. But
/// it is also how a distributed run becomes a local one with nothing to notice
/// -- the fleet demo named its workers `PrefillWorker` and `DecodeWorker` while
/// the program dispatched to `GPU[0]` and `GPU[1]`, so every dispatch fell
/// through to local. The run then reported two GPUs in use and identical output
/// from both halves. Both were true. Neither meant what it appeared to.
///
/// So: one line, unconditionally -- not behind VX_DISPATCH_VERBOSE, because the
/// runs that need it most are the ones nobody thought to instrument. Once per
/// distinct placement, because a decode loop would otherwise print it per
/// token.
///
/// `VX_FLEET_STRICT=1` makes it fatal, for a run whose whole purpose is that
/// every placement is remote. A demo harness should set it.
inline void vx_routing_note_unlisted(int32_t topology_id, const char *name) {
  // The host is not a worker and is never in a manifest: `topology_dispatch_id`
  // gives the CPU 0 by construction, so "topology 0 is absent" is a restatement
  // of what placing something on the host means. Saying it would be noise on
  // every run, and aborting on it under strict mode would reject the ordinary
  // case of a program that places some regions on a device and leaves the rest
  // where they are.
  if (topology_id == 0) {
    return;
  }

  static const bool strict = [] {
    const char *v = getenv("VX_FLEET_STRICT");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();

  static int32_t seen[64];
  static size_t seen_count = 0;
  static std::mutex mu;

  bool first = false;
  {
    std::lock_guard<std::mutex> guard(mu);
    bool found = false;
    for (size_t i = 0; i < seen_count; ++i) {
      if (seen[i] == topology_id) {
        found = true;
        break;
      }
    }
    if (!found) {
      first = true;
      if (seen_count < sizeof(seen) / sizeof(seen[0])) {
        seen[seen_count++] = topology_id;
      }
    }
  }
  if (!first) {
    return;
  }

  fprintf(stderr,
          "[Vx remote] topology %d%s%s%s is not in the manifest; it runs on "
          "this machine\n",
          (int)topology_id, name ? " (" : "", name ? name : "",
          name ? ")" : "");
  if (strict) {
    fprintf(stderr,
            "[Vx remote] FATAL: VX_FLEET_STRICT is set, and this run was "
            "supposed to place every region on a worker.\n");
    abort();
  }
}

/// The worker a topology id names, or NULL for "local".
inline const vx_manifest_entry *vx_routing_worker(int32_t topology_id) {
  const vx_manifest &m = vx_routing_manifest();
  const vx_manifest_entry *w = NULL;
  if (vx_manifest_classify(&m, NULL, topology_id, &w) == VX_PLACED_UNLISTED) {
    vx_routing_note_unlisted(topology_id, NULL);
  }
  return w;
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
  const char *name = vx_payload_field(payload, size, "toponame=");
  int32_t id = vx_payload_topology(payload, size);
  const vx_manifest_entry *w = NULL;
  if (vx_manifest_classify(&m, name, id, &w) == VX_PLACED_UNLISTED) {
    // The name is worth carrying into the message: `topology 1113` is a hash
    // and cannot be read back, so a mismatch between what the program placed
    // and what the manifest named is only legible with the spelling.
    vx_routing_note_unlisted(id, name);
  }
  return w;
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

  /* A result the kernel *allocates* cannot be collected across the wire yet.
     `outkind=buffer` names a buffer that exists here, so the worker fills the
     copy it was staged and vx_remote_dispatch fetches it back. `outkind=slot`
     means the plugin allocates the result on the far side and publishes a
     descriptor for it, and that descriptor names memory in another process:
     nothing in this protocol brings it home.

     Declining is the point. Without it the dispatch is sent, the worker runs
     it, and no result comes back -- leaving the caller's tensor holding
     whatever it held before. A GEMM benchmark measured this as 560 dispatches
     served and every answer zero, with a success status and no diagnostic: the
     initialised value, returned as if it were the product. Running it here
     instead costs the distribution and keeps the arithmetic, which is the
     trade this backend makes everywhere else. */
  {
    const char *outkind = vx_payload_field(payload, payload_size, "outkind=");
    if (!outkind || strcmp(outkind, "buffer") != 0) {
      static int said = 0;
      if (!said) {
        said = 1;
        fprintf(stderr,
                "[Vx remote] %s results cannot cross a worker boundary; "
                "running these dispatches locally instead (#348)\n",
                outkind ? outkind : "unnamed");
      }
      return 0;
    }
  }

  static vx_wire_arg args[64];
  if (num_args > 64) {
    return 0;
  }
  size_t scratch_len = 0;
  uint8_t *scratch = vx_routing_scratch(&scratch_len);
  if (vx_remote_dispatch(vx_routing_fd(w), payload, (uint64_t)payload_size,
                         device_args, arg_tags, num_args, args, scratch,
                         scratch_len)) {
    return 1;
  }

  /* Everything above this point declines *before* touching the wire, and
     returning 0 there means "run it here", which is safe. Failing here does
     not mean that. By now operands have been staged onto the worker and the
     handles the caller holds name memory in another process, so the local path
     cannot run this dispatch -- it would stage from a handle, and the fault
     lands in cudaMemcpy2D with nothing in the backtrace to say why.

     This is how a worker running out of address space presented itself: a
     TRANSFER returned handle 0, encoding failed, dispatch reported failure,
     the caller obligingly ran it locally, and the program died on the *other*
     machine with a segmentation fault. Refusing to pretend is the difference
     between a diagnosis and a mystery. */
  fprintf(stderr,
          "[Vx remote] FATAL: dispatch to %s (%s:%d) failed after its operands "
          "were staged there.\n"
          "            Running it locally is not possible -- the operands are "
          "on the worker -- and continuing would fault (#348).\n",
          w->name, w->host, w->port);
  abort();
}

/// Returns 1 and sets `*out` when either end of a peer transfer is remote.
///
/// A movement between two devices becomes, across machines, a read from one and
/// a write to the other -- the same shape the CUDA backend already falls back
/// to when no P2P path exists, which is why `cudaMemcpyPeer` is allowed to
/// stage through the host. Worker to worker directly is the optimisation, and
/// needs a message this protocol does not have.
///
/// Without this the local path is reached with a handle, `is_device_ptr` says
/// it is not device memory, and it memcpys from a non-canonical address. That
/// faults immediately rather than reading something -- which is the whole
/// reason handles are minted non-canonical -- but it is still the wrong thing
/// to do, and llama2.vx's `handoff_kv` is exactly where it happens.
inline int vx_routing_try_peer(void *src, uint32_t src_topology_id,
                               uint32_t dst_topology_id, size_t bytes,
                               void **out) {
  const vx_manifest_entry *sw = vx_routing_worker((int32_t)src_topology_id);
  const vx_manifest_entry *dw = vx_routing_worker((int32_t)dst_topology_id);
  if (!sw && !dw) {
    return 0;
  }

  size_t scratch_len = 0;
  uint8_t *scratch = vx_routing_scratch(&scratch_len);
  /* The staging buffer carries a message, so the bytes have to fit beside its
     header rather than exactly fill it. */
  if (bytes + 4096 > scratch_len) {
    fprintf(stderr, "[Vx remote] FATAL: a %zu-byte handoff exceeds staging\n",
            bytes);
    abort();
  }

  /* Read the source into host memory, wherever it is. */
  if (sw && vx_remote_addr_is_handle((uint64_t)(uintptr_t)src)) {
    if (!vx_remote_fetch(vx_routing_fd(sw), (uint64_t)(uintptr_t)src, scratch,
                         (uint64_t)bytes)) {
      fprintf(stderr, "[Vx remote] FATAL: %s could not return the handoff\n",
              sw->name);
      abort();
    }
  } else {
    memcpy(scratch, src, bytes);
  }

  /* And write it wherever the destination is. */
  if (dw) {
    void *handle = nullptr;
    if (!vx_routing_try_alloc(bytes, scratch, dst_topology_id, &handle)) {
      return 0;
    }
    *out = handle;
  } else {
    void *dst = malloc(bytes);
    if (dst) {
      memcpy(dst, scratch, bytes);
    }
    *out = dst;
  }
  return 1;
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
