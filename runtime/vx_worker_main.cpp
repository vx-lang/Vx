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
#include <time.h>
#include <unistd.h>

namespace {

/// Which of this machine's devices to place work on. A worker owns one, which
/// is what makes it a worker rather than a scheduler.
int32_t g_topology = VX_TOPO_GPU_BASE;
/// Narration, from the flag or from the same variable the plugin reads.
///
/// These were separate, and the difference is invisible until it matters: a
/// harness exporting VX_DISPATCH_VERBOSE got a log full of `[Vx CUDA]` lines
/// from the plugin linked into this process and not one `[Vx worker]` line, so
/// a check for the messages *served* found none and reported that the run had
/// never left the host. It had; 77064 device operations were in the same file.
bool g_verbose = [] {
  const char *v = getenv("VX_DISPATCH_VERBOSE");
  return v && v[0] != '\0' && strcmp(v, "0") != 0;
}();
uint32_t g_worker_id = 1;

/* Sized for a generation rather than a demo. Every dispatch stages its
   activations and frees them after, and `vx_remote_table_free` marks a region
   dead without reusing its slot -- deliberately, so a stale handle resolves to
   something dead rather than to whatever was allocated next. That makes the
   table grow with the number of dispatches: llama2 at 64 tokens is about 5600
   stagings, and 4096 entries ran out mid-generation.
   Reusing a slot once no handle can name it is the real fix and needs a
   generation check on resolve; 256K entries is 12 MB and buys a run long
   enough not to need it yet. */
vx_remote_region g_regions[262144];
vx_remote_table g_table;

/* Sized for llama2's largest staged blob rather than for a token: a projection
   weight arrives in one TRANSFER. */
uint8_t g_body[64u << 20];
/* Wire traffic this worker has served, by message kind (1..4). Reported when
   the host disconnects; see the counting site in the message loop. */
uint64_t g_msg_count[5] = {0, 0, 0, 0, 0};
uint64_t g_msg_bytes[5] = {0, 0, 0, 0, 0};

/* How long the device spent, which is a different question from how long the
   program took and the only one this side of the network can answer.
   Accumulated per (m,n,k) so a run that changes shape does not average across
   shapes. */
struct dispatch_timing {
  int64_t m, n, k;
  uint64_t count;
  double best;  /* seconds */
  double total; /* seconds */
};
dispatch_timing g_timing[8];
size_t g_timing_count = 0;

double now_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void note_dispatch_time(const vx_gemm_plan *plan, double secs) {
  for (size_t i = 0; i < g_timing_count; ++i) {
    if (g_timing[i].m == plan->m && g_timing[i].n == plan->n &&
        g_timing[i].k == plan->k) {
      ++g_timing[i].count;
      g_timing[i].total += secs;
      if (secs < g_timing[i].best) {
        g_timing[i].best = secs;
      }
      return;
    }
  }
  if (g_timing_count == sizeof(g_timing) / sizeof(g_timing[0])) {
    return;
  }
  g_timing[g_timing_count].m = plan->m;
  g_timing[g_timing_count].n = plan->n;
  g_timing[g_timing_count].k = plan->k;
  g_timing[g_timing_count].count = 1;
  g_timing[g_timing_count].best = secs;
  g_timing[g_timing_count].total = secs;
  ++g_timing_count;
}

/* Best and mean, because the two say different things and quoting one is how a
   GEMM number becomes wrong. The first dispatch of a process builds the CUDA
   context and the cuBLAS handle and takes milliseconds that have nothing to do
   with the arithmetic; it lands in the mean and not in the best. A run whose
   best and mean are far apart is a run that was still warming up. */
void report_timing() {
  for (size_t i = 0; i < g_timing_count; ++i) {
    const dispatch_timing *t = &g_timing[i];
    /* 2mnk: one multiply and one add per element of the product. */
    const double flops = 2.0 * (double)t->m * (double)t->n * (double)t->k;
    const double mean = t->total / (double)t->count;
    fprintf(stderr,
            "[Vx worker] gemm %lldx%lldx%lld  x%llu  best %.3f ms (%.1f "
            "GFLOP/s)  mean %.3f ms (%.1f GFLOP/s)\n",
            (long long)t->m, (long long)t->n, (long long)t->k,
            (unsigned long long)t->count, t->best * 1e3, flops / t->best / 1e9,
            mean * 1e3, flops / mean / 1e9);
  }
  g_timing_count = 0;
}

/* One line per kind when the host goes away, so a harness can read the cost of
   a run without parsing a log. Unconditional rather than behind --verbose: a
   run whose traffic nobody measured is how "about six round trips" survived as
   an estimate. Silent when nothing was served, so an idle worker stays quiet.
 */
void report_traffic() {
  static const char *kNames[5] = {"", "TRANSFER", "DISPATCH", "FREE", "FETCH"};
  uint64_t total = 0;
  for (int i = 1; i <= 4; ++i) {
    total += g_msg_count[i];
  }
  if (total == 0) {
    return;
  }
  fprintf(stderr,
          "[Vx worker] served %llu message(s):", (unsigned long long)total);
  for (int i = 1; i <= 4; ++i) {
    if (g_msg_count[i] != 0) {
      fprintf(stderr, " %s=%llu (%llu B)", kNames[i],
              (unsigned long long)g_msg_count[i],
              (unsigned long long)g_msg_bytes[i]);
    }
  }
  fprintf(stderr, "\n");
}

/* A dispatch reply is a status and a few handles, but a FETCH reply is bulk:
   llama2's KV handoff reads back n_layers x seq_len x kv_dim x 4 bytes in one
   message, 442 KB at 64 tokens and far more at a real context length. Sized for
   the dispatch case at 64 KiB, the worker refused every handoff it was asked
   for -- correctly, since it will not invent bytes it cannot hold, but the
   refusal read as "could not return the handoff" with nothing to say why. */
uint8_t g_reply[64u << 20];

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
    /* Which one, and which handle. vx_agent.h refuses without naming anything
       on purpose -- it cannot substitute a plausible region -- and says the
       caller will report it. This is that report, and until it existed the
       message was "an argument named no live region" with no way to tell an
       operand that was never sent from one that was freed too early. */
    log_line("[Vx worker] refused: an argument named no live region\n");
    for (int64_t i = 0; i < d->num_args; ++i) {
      vx_remote_ref ref;
      if (args[i].handle != 0 &&
          !vx_remote_resolve(&g_table, args[i].handle, &ref)) {
        log_line("[Vx worker]   arg %lld: handle %llx is not one this worker "
                 "minted or still holds\n",
                 (long long)i, (unsigned long long)args[i].handle);
      }
    }
    return 0;
  }

  /* Asked before dispatching, because the plugin's own answer to "I cannot
     route this" is to look for the outlined kernel, which is not on this
     machine. Deciding here keeps that abort from ever being reached.
   *
   * Two ways to be routable now. A classified matmul decodes to a plan and
   * goes to a vendor GEMM. Anything else is routable if the payload carries
   * the kernel: the compiler compiles every self-contained device region to
   * PTX and puts the image in the blob (#251), so the plugin can load and
   * launch it here rather than hunting for a host symbol that only exists in
   * the program that sent this.
   *
   * Zeroed rather than left indeterminate, because the fields below are read
   * on the timing and slot-result paths and only the GEMM route fills them. */
  memset(&plan, 0, sizeof(plan));
  const bool routed_as_gemm =
      vx_gemm_plan_decode(d->payload, (size_t)d->payload_len, device_args,
                          arg_tags, d->num_args, &plan);
  const bool carries_kernel =
      vx_payload_field(d->payload, (size_t)d->payload_len, "image=") != nullptr;
  if (!routed_as_gemm && !carries_kernel) {
    log_line("[Vx worker] refused: not a kernel this worker can route, and no "
             "device image came with it\n");
    return 0;
  }

  /* The topology id means two different things on the two sides of the wire,
     and the worker is where they have to be told apart.
   *
   * To the host, `topo=501` names *which machine* -- the manifest resolves it
   to
   * this one. To a plugin it names *which local device*, and
   * `vx_plugin_dispatch_async` reads it straight out of the payload and calls
   * `select_device` with it. A worker whose machine has one GPU then aborts
   with
   * "launch targets GPU 1, but this machine has 1", which is correct behaviour
   * answering the wrong question.
   *
   * So the payload's topology is rewritten, in place, to the device this worker
   * was told to use. Digits are padded with leading zeros rather than the field
   * being resized: the blob is NUL-separated and moving its tail would shift
   * every entry after it, and `strtol` reads "0500" as 500. */
  {
    char *topo =
        (char *)vx_payload_field(d->payload, (size_t)d->payload_len, "topo=");
    if (topo) {
      size_t width = strlen(topo);
      char local[32];
      snprintf(local, sizeof(local), "%d", g_topology);
      size_t n = strlen(local);
      if (n <= width) {
        memset(topo, '0', width - n);
        memcpy(topo + width - n, local, n);
      } else {
        log_line("[Vx worker] cannot retarget topo= (%s needs %zu > %zu)\n",
                 local, n, width);
        return 0;
      }
    }
  }

  /* Timed here because nowhere else can. A dispatch driven from another
     machine spends its wall clock on the link -- 172 ms of round trip against
     tens of microseconds of A100 -- so the host's clock measures the network
     and says nothing about the device. This one is on the far side of the
     network, immediately around the call, and `vx_plugin_dispatch_async`
     synchronises before returning, so what it brackets is the GEMM (#321).
     Still not a kernel timer: it includes the plugin's own argument handling
     and, on the first call of a process, the CUDA context and the cuBLAS
     handle -- which is why the summary reports the best of the run and not the
     mean. */
  const double t0 = now_seconds();
  vx_plugin_dispatch_async(d->payload, (size_t)d->payload_len, device_args,
                           arg_tags, d->num_args);
  const double elapsed = now_seconds() - t0;
  /* Only a GEMM has an (m,n,k) to accumulate against, and only a GEMM has a
     FLOP count that means anything. Timing an emitted kernel under the same
     heading would report GFLOP/s for a fused softmax computed from a plan that
     was never decoded. */
  if (routed_as_gemm) {
    note_dispatch_time(&plan, elapsed);
  }

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

  if (routed_as_gemm) {
    log_line("[Vx worker] DISPATCH %lldx%lldx%lld -> %lld result(s)\n",
             (long long)plan.m, (long long)plan.n, (long long)plan.k,
             (long long)*num_results);
  } else {
    log_line("[Vx worker] DISPATCH %s from its own image -> %lld result(s)\n",
             (const char *)d->payload, (long long)*num_results);
  }
  return 1;
}

int serve(int fd) {
  /* Per connection, not per process. A worker that serves two runs in a row
     reported the second as the sum of both -- 64 messages then "78", which is
     64 + 14 and reads as a run that sent more than it did. */
  for (int i = 0; i < 5; ++i) {
    g_msg_count[i] = 0;
    g_msg_bytes[i] = 0;
  }

  for (;;) {
    uint32_t type = 0;
    uint64_t len = 0;
    int rc = vx_transport_recv(fd, &type, g_body, sizeof(g_body), &len);

    if (rc == VX_TRANSPORT_EOF) {
      log_line("[Vx worker] host disconnected\n");
      report_traffic();
      report_timing();
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
      report_traffic();
      report_timing();
      return 1;
    }

    vx_wire_reader r;
    vx_wire_reader_init(&r, g_body, (size_t)len);

    // Exact per-kind accounting. Round trips are the fleet's unit of cost --
    // each one is a full latency, and on a real network that dominates a small
    // operand's bytes by orders of magnitude -- so how many a dispatch takes
    // has to be measurable rather than inferred from which log lines happen to
    // exist. TRANSFER and DISPATCH were logged; FETCH and FREE were not, which
    // made "about six round trips per dispatch" an estimate nobody could check.
    if (type >= 1 && type <= 4) {
      g_msg_count[type] += 1;
      g_msg_bytes[type] += (uint64_t)len;
    }

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
      log_line("[Vx worker] FREE handle %llx\n", (unsigned long long)handle);
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
