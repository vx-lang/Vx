//===- vx_worker_main.cpp - A machine that serves dispatches ----*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The process that runs on a worker machine: listen, and serve the four
// messages a remote dispatch is made of (#348).
//
// **It contains no vendor code and no dispatch logic.** Every message turns
// into the corresponding call on the plugin ABI -- the same
// `vx_plugin_alloc_and_transfer`, `vx_plugin_dispatch_async`,
// `vx_plugin_transfer_device_to_host` and `vx_plugin_free` a local program
// makes -- so whatever this is linked against decides what the worker is. Built
// against runtime/cuda_dispatch.cpp it is a GPU worker; against
// runtime/host_dispatch.cpp it is a CPU one; and the wire between them does not
// change. That is the same "one operation, a different provider" the client
// side has, seen from the other end.
//
// The one judgement it makes is whether a dispatch is *routable*, and it makes
// that with vx_dispatch_plan.h -- the decoder a local plugin uses. A kernel the
// plugin cannot recognise falls back locally to `dlsym` for the outlined
// function, which a worker does not have: the artifact was never shipped here,
// and `vx_plugin_dispatch_async` would abort looking for it. So an unroutable
// dispatch is *refused*, and the host runs it itself. That is exactly what
// happens today on one machine when a plugin declines a kernel, which is why
// the demo path needs no artifact shipping at all: every projection in
// llama2.vx is a classified matmul, and a classified matmul never touches the
// outlined kernel.
//
// Usage:
//   vx-worker --port 9001 [--topology 500] [--verbose]
//
//===----------------------------------------------------------------------===//

#include "../include/vx_hardware_runtime.h"
#include "vx_agent.h"
#include "vx_dispatch_plan.h"
#include "vx_remote_region.h"
#include "vx_transport.h"
#include "vx_wire.h"

#include <arpa/inet.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

/// Which of this machine's devices to place work on. A worker owns one, which
/// is what makes it a worker rather than a scheduler.
int32_t g_topology = VX_TOPO_GPU_BASE;
bool g_verbose = false;
uint32_t g_worker_id = 1;

vx_remote_region g_regions[4096];
vx_remote_table g_table;

/* Sized for llama2's largest staged blob rather than for a token: a projection
   weight arrives in one TRANSFER. */
uint8_t g_body[64u << 20];
uint8_t g_reply[64u << 10];

void log_line(const char *fmt, ...) {
  if (!g_verbose) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

/// TRANSFER: put the bytes where this worker's device can read them, and name
/// the result.
uint64_t serve_transfer(const vx_wire_transfer *t) {
  void *dev = vx_plugin_alloc_and_transfer(
      (size_t)t->nbytes, (void *)(uintptr_t)t->bytes, (uint32_t)g_topology);
  if (!dev) {
    return 0;
  }
  uint64_t handle =
      vx_remote_table_alloc(&g_table, g_worker_id, t->nbytes, t->dtype, dev);
  if (handle == 0) {
    vx_plugin_free(dev, (uint32_t)g_topology);
    return 0;
  }
  log_line("[Vx worker] TRANSFER %llu bytes -> handle %llx\n",
           (unsigned long long)t->nbytes, (unsigned long long)handle);
  return handle;
}

/// DISPATCH: rebuild the arguments as a local dispatch's, hand them to the
/// plugin, and name whatever it published.
///
/// Returns 0 when the dispatch was not served, which the host must treat as
/// "run it yourself" rather than as an error -- an unroutable kernel is a
/// performance outcome locally and has to stay one here.
int serve_dispatch(const vx_wire_dispatch *d, const vx_wire_arg *args,
                   vx_wire_result *results, int64_t *num_results) {
  void *device_args[16];
  void *desc_ptrs[16];
  int32_t arg_tags[16];
  uint8_t descriptors[16 * 256];
  vx_agent_args storage;
  vx_gemm_plan plan;

  if (d->num_args > 16) {
    return 0;
  }
  storage.device_args = device_args;
  storage.desc_ptrs = desc_ptrs;
  storage.arg_tags = arg_tags;
  storage.descriptors = descriptors;
  storage.descriptors_capacity = sizeof(descriptors);

  if (!vx_agent_rebuild_args(&g_table, args, d->num_args, &storage)) {
    log_line("[Vx worker] refused: an argument named no live region\n");
    return 0;
  }

  /* Asked before dispatching, because the plugin's own answer to "I cannot
     route this" is to look for the outlined kernel, which is not on this
     machine. Deciding here keeps that abort from ever being reached. */
  if (!vx_gemm_plan_decode(d->payload, (size_t)d->payload_len, device_args,
                           arg_tags, d->num_args, &plan)) {
    log_line("[Vx worker] refused: not a kernel this worker can route\n");
    return 0;
  }

  vx_plugin_dispatch_async(d->payload, (size_t)d->payload_len, device_args,
                           arg_tags, d->num_args);

  /* A slot result was allocated by the plugin and is unknown to the region
     table until now. Its size is the plan's, which is why the plan was decoded
     rather than merely consulted. */
  *num_results = 0;
  for (int64_t i = 0; i < d->num_args; ++i) {
    /* A slot that arrived *holding* a buffer is not a publication target: the
       plugin filled what it already named, the way `outkind=buffer` means. Only
       a slot that arrived empty is storage something was published into.
       Reporting the first kind as a result made the host overwrite its own
       descriptor with a handle, and the read-back that followed wrote into it.
     */
    if (!VX_ABI_IS_SLOT(args[i].tag) || args[i].handle != 0) {
      continue;
    }
    const void *outer = desc_ptrs[i];
    const void *inner = vx_memref_aligned(outer);
    void *published = vx_memref_aligned(inner);
    const int64_t *sizes = vx_memref_sizes(inner);
    uint64_t bytes = (uint64_t)plan.m * (uint64_t)plan.n *
                     (uint64_t)vx_dtype_bytes(plan.dtype);
    uint64_t handle;

    if (!published) {
      log_line("[Vx worker] refused: a slot was never published\n");
      return 0;
    }
    handle = vx_remote_table_alloc(&g_table, g_worker_id, bytes, plan.dtype,
                                   published);
    if (handle == 0) {
      return 0;
    }
    results[*num_results].handle = handle;
    results[*num_results].rank = (int32_t)VX_ABI_RANK(args[i].tag);
    for (int32_t j = 0; j < results[*num_results].rank; ++j) {
      results[*num_results].sizes[j] = sizes[j];
    }
    ++(*num_results);
  }

  log_line("[Vx worker] DISPATCH %lldx%lldx%lld -> %lld result(s)\n",
           (long long)plan.m, (long long)plan.n, (long long)plan.k,
           (long long)*num_results);
  return 1;
}

int serve(int fd) {
  for (;;) {
    uint32_t type = 0;
    uint64_t len = 0;
    int rc = vx_transport_recv(fd, &type, g_body, sizeof(g_body), &len);

    if (rc == VX_TRANSPORT_EOF) {
      log_line("[Vx worker] host disconnected\n");
      return 0;
    }
    if (rc == VX_TRANSPORT_TOO_LARGE) {
      fprintf(stderr, "[Vx worker] a message exceeded the %zu-byte buffer\n",
              sizeof(g_body));
      /* Drained by the transport, so answering keeps the stream in step. */
      vx_transport_send(fd, type, nullptr, 0);
      continue;
    }
    if (rc != VX_TRANSPORT_OK) {
      fprintf(stderr, "[Vx worker] transport failure; dropping the host\n");
      return 1;
    }

    vx_wire_reader r;
    vx_wire_reader_init(&r, g_body, (size_t)len);

    switch (type) {
    case VX_WIRE_TRANSFER: {
      vx_wire_transfer t;
      uint64_t handle = 0;
      if (vx_wire_get_transfer(&r, &t)) {
        handle = serve_transfer(&t);
      }
      vx_wire_writer w;
      vx_wire_writer_init(&w, g_reply, sizeof(g_reply));
      vx_wire_put_u64(&w, handle);
      if (vx_transport_send(fd, VX_WIRE_TRANSFER, g_reply, w.len) !=
          VX_TRANSPORT_OK) {
        return 1;
      }
      break;
    }
    case VX_WIRE_DISPATCH: {
      vx_wire_dispatch d;
      vx_wire_arg args[16];
      vx_wire_result results[8];
      int64_t num_results = 0;
      int ok = vx_wire_get_dispatch(&r, &d) && d.num_args <= 16;
      for (int64_t i = 0; ok && i < d.num_args; ++i) {
        ok = vx_wire_get_arg(&r, &args[i]);
      }
      if (ok) {
        ok = serve_dispatch(&d, args, results, &num_results);
      }
      vx_wire_writer w;
      vx_wire_writer_init(&w, g_reply, sizeof(g_reply));
      vx_wire_put_results(&w, ok ? 0 : -1, results, ok ? num_results : 0);
      if (vx_transport_send(fd, VX_WIRE_DISPATCH, g_reply, w.len) !=
          VX_TRANSPORT_OK) {
        return 1;
      }
      break;
    }
    case VX_WIRE_FETCH: {
      uint64_t handle = 0, nbytes = 0;
      vx_remote_ref ref;
      if (!vx_wire_get_fetch(&r, &handle, &nbytes) ||
          !vx_remote_resolve(&g_table, handle, &ref) ||
          nbytes > ref.region->size - ref.offset || nbytes > sizeof(g_reply)) {
        /* Nothing, rather than bytes invented for a handle that named nothing
           or a length that ran past what it named. */
        if (vx_transport_send(fd, VX_WIRE_FETCH, nullptr, 0) !=
            VX_TRANSPORT_OK) {
          return 1;
        }
        break;
      }
      vx_plugin_transfer_device_to_host((char *)ref.region->remote + ref.offset,
                                        g_reply, (size_t)nbytes,
                                        (uint32_t)g_topology);
      if (vx_transport_send(fd, VX_WIRE_FETCH, g_reply, nbytes) !=
          VX_TRANSPORT_OK) {
        return 1;
      }
      break;
    }
    case VX_WIRE_FREE: {
      uint64_t handle = 0;
      vx_remote_ref ref;
      if (vx_wire_get_free(&r, &handle) &&
          vx_remote_resolve(&g_table, handle, &ref) && ref.offset == 0) {
        vx_plugin_free(ref.region->remote, (uint32_t)g_topology);
        vx_remote_table_free(&g_table, handle);
      }
      vx_transport_send(fd, VX_WIRE_FREE, nullptr, 0);
      break;
    }
    default:
      fprintf(stderr, "[Vx worker] unknown message type %u\n", type);
      return 1;
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  int port = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--topology") == 0 && i + 1 < argc) {
      g_topology = (int32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--worker-id") == 0 && i + 1 < argc) {
      g_worker_id = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "--verbose") == 0) {
      g_verbose = true;
    } else {
      fprintf(stderr,
              "usage: %s --port N [--topology T] [--worker-id W] [--verbose]\n",
              argv[0]);
      return 2;
    }
  }
  if (port <= 0 || port > 65535) {
    fprintf(stderr, "error: --port is required and must be 1..65535\n");
    return 2;
  }
  if (g_worker_id == 0 || g_worker_id > VX_REMOTE_WORKER_MAX) {
    fprintf(stderr, "error: --worker-id must be 1..%d\n", VX_REMOTE_WORKER_MAX);
    return 2;
  }

  /* A host that goes away mid-reply must not take the worker with it. */
  signal(SIGPIPE, SIG_IGN);

  vx_remote_table_init(&g_table, g_regions,
                       sizeof(g_regions) / sizeof(g_regions[0]));

  int listener = socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) {
    perror("socket");
    return 1;
  }
  int one = 1;
  setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons((uint16_t)port);
  if (bind(listener, (sockaddr *)&addr, sizeof(addr)) != 0) {
    perror("bind");
    return 1;
  }
  if (listen(listener, 4) != 0) {
    perror("listen");
    return 1;
  }

  fprintf(stderr, "[Vx worker] listening on %d, topology %d, worker id %u\n",
          port, g_topology, g_worker_id);

  /* One host at a time, served to completion. A worker owns a device, and two
     hosts dispatching onto it concurrently would interleave on that device
     with nothing here sequencing them. */
  for (;;) {
    int fd = accept(listener, nullptr, nullptr);
    if (fd < 0) {
      perror("accept");
      continue;
    }
    int one_nd = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one_nd, sizeof(one_nd));
    fprintf(stderr, "[Vx worker] host connected\n");
    serve(fd);
    close(fd);
  }
}
